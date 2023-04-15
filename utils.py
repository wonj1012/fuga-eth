
import hashlib
import io
import os
import boto3
from dotenv import load_dotenv
import torch

# load .env
load_dotenv()

S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
S3_SCRETE_KEY = os.environ.get('S3_SCRETE_KEY')
BUCKET_NAME = 'fugaeth'


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
        raise("hash value is not matched")

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