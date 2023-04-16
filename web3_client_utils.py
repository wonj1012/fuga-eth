
import hashlib
import io
import os
import boto3
from dotenv import load_dotenv
import torch
import numpy as np

# load .env
load_dotenv()

S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
S3_SCRETE_KEY = os.environ.get('S3_SCRETE_KEY')
HTTP_PROVIDER = os.environ.get('HTTP_PROVIDER')

# MNEMONIC = os.environ.get('MNEMONIC')
BUCKET_NAME = 'fugaeth'

# def get_private_key(MNEMONIC):
#     w3 = web3.Web3()
#     w3.eth.account.enable_unaudited_hdwallet_features()
#     account = w3.eth.account.from_mnemonic(MNEMONIC, account_path="m/44'/60'/0'/0/0")
#     return account.key

# PRIVATE_KEY = get_private_key(MNEMONIC)

def send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY):
    """
    Sends a transaction to the blockchain.
    """

    # create the transaction
    tx = {
        'chainId': w3.eth.chainId,
        'from': PUBLIC_KEY,
        'to': contract.address,
        'gas': 2000000,
        'gasPrice': w3.eth.gasPrice,
        'nonce' :w3.eth.getTransactionCount(PUBLIC_KEY),
        'data': function._encode_transaction_data()
    }

    # sign the transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)

    # send the transaction
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    # wait for the transaction to be mined
    tx_recipt = w3.eth.waitForTransactionReceipt(tx_hash)

    return tx_recipt

def read_transaction(w3, contract, function_name, PUBLIC_KEY, PRIVATE_KEY):
    function = getattr(contract.functions, function_name)()
    tx_receipt = send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)
    if function_name=='getConfig':
        event_logs = contract.events.getConfigMessage().processReceipt(tx_receipt)
    elif function_name=='getClient':
        event_logs = contract.events.getClientMessage().processReceipt(tx_receipt)
    elif function_name=='FitIns':
        event_logs = contract.events.FitInsMessage().processReceipt(tx_receipt)
    elif function_name=='EvaluateIns':
        event_logs = contract.events.EvaluateInsMessage().processReceipt(tx_receipt)
    return event_logs[0]['args']

def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SCRETE_KEY
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def make_hash(params):
    # Concatenate all arrays in the list of parameters
    concatenated_array = np.concatenate([param.flatten() for param in params])

    # Convert the concatenated array to bytes and calculate the hash
    hash_bytes = hashlib.sha256(concatenated_array.tobytes()).digest()

    # Convert the hash to a string and return it
    return hash_bytes.hex()

def read_model(model_hash):
    with open(f'./models/{model_hash}.bin', 'rb') as f:
        buffer = io.BytesIO(f.read())
        param = torch.load(buffer)
    return param

def check_model(model_hash):
    param = read_model(model_hash)
    if not make_hash(param)==model_hash:
        print("hash value is not matched")
        return False
    else:
        return True

def upload_model(s3, parameters_prime, model_hash):
    buffer = io.BytesIO()
    torch.save(parameters_prime, buffer)
    serialized_model = buffer.getvalue()
    with open(f'./models/{model_hash}.bin', 'wb') as f:
        f.write(serialized_model)
    # specify the bucket name and object key
    object_key = f'models/{model_hash}.bin'

    # upload the file to S3
    with open(f'./models/{model_hash}.bin', 'rb') as f:
        s3.upload_fileobj(f, BUCKET_NAME, object_key)

        
def aggregate_fit(client, model_hashes, num_samples, scores):
    params = []
    for model_hash in model_hashes:
        params.append(read_model(model_hash)) 
        
    # Normalize the evaluation scores
    sum_scores = sum(scores)
    norm_scores = [score / sum_scores for score in scores]

    # Combine normalized evaluation scores with the dataset portion
    sum_samples = sum(num_samples)
    combined_weights = [norm_score * (num_sample / sum_samples) for norm_score, num_sample in zip(norm_scores, num_samples)]

    # Calculate the weighted model updates
    weighted_updates = [np.multiply(w, update) for w, update in zip(combined_weights, params)]
    new_params = [sum(updates) for updates in zip(*weighted_updates)]

    return client.fit(new_params)    
