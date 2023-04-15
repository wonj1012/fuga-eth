from typing import Callable, Dict, Tuple
from dotenv import load_dotenv
import time
import numpy as np
import web3
from logging import INFO

from utils import *

# load .env
load_dotenv()
PUBLIC_KEY = os.environ.get('PUBLIC_KEY')
PRIVATE_KEY = os.environ.get('PRIVATE_KEY')

def listen_for_event(contract, event_name):
    # create a filter to listen for the specified event
    event_filter = contract.events[event_name].createFilter(fromBlock='latest')

    while True:
        # check if any new events have been emitted
        for event in event_filter.get_new_entries():
            # if the specified event has been emitted, return its message
            if event.event == event_name:
                yield event.args
                return
        # wait for new events
        time.sleep(60)

def web3_connection(s3, w3, contract) -> Tuple[Callable[[], Dict], Callable[[Dict], None]]:
    """
    Creates a connection to the blockchain and returns a function to receive messages and a function to send messages.    
    """
    # create a filter to listen for the specified event
    web3_message_iterator = listen_for_event(contract, "ServerMessage")

    # create a function that returns the next message
    receive: Callable[[], Dict] = lambda: next(web3_message_iterator)
    send: Callable[[Dict], None] = lambda msg: handle_send(s3, w3, contract, msg, PUBLIC_KEY, PRIVATE_KEY)

    return (receive, send)


def handle_receive(client, s3, contract, msg):
    """
    Handles a received message.
    Returns a tuple containing the response message, the number of samples and a boolean indicating whether the client should shut down.
    """
    # check the field of the message
    field = msg['field']

    # if the message is a request for the FitIns, aggregate the model and return the result after fit
    if field == "FitIns":
        # receive the FitIns data
        response = contract.functions.FitIns().call()
        
        model_hashes = response[0]
        num_samples =response[1]
        scores = response[2]
        batch_size = response[3]
        learning_rate = response[4]
        local_epochs = response[5]
        config = {'batch_size': batch_size, 'lr' : learning_rate ,'local_epochs': local_epochs}

        for model_hash in model_hashes:
            object_key = f'models/{model_hash}.bin'
            file_path = object_key

            s3.download_file(BUCKET_NAME, object_key, file_path)

            # check hash value
            check_model(model_hash)

        # aggregate the model
        # parameters_prime, num_examples_train, results = aggregate_fit(client, model_hashes, num_samples, scores, config)
        fitres = aggregate_fit(client, model_hashes, num_samples, scores, config)

        # return the result of fit
        return {'field':'FitRes', 'data': fitres},0, True
    
    # if the message is a request for the EvaluateIns, evaluate the model and return the result
    elif field == "EvaluateIns":
        # evaluate the model on client
        response = contract.functions.EvaluateIns().call()

        model_hashes = response[0]
        batch_size = response[1]
        val_steps = response[2]
        config = {'batch_size': batch_size, 'val_steps': val_steps}
        evalres = {}

        for model_hash in model_hashes:
            object_key = f'models/{model_hash}.bin'
            file_path = object_key

            s3.download_file(BUCKET_NAME, object_key, file_path)

            # check hash value
            check_model(model_hash)
            param = read_model(model_hash)

            # evaluate the model
            loss, _, _ = client.evaluate(param,config)
            evalres[model_hash] = loss

        return {'field':'EvaluateRes', 'data':evalres}, 0 , True


def handle_send(s3, w3, contract, msg, sender_address, sender_private_key):
    """
    Handles a message to be sent.
    Returns the transaction receipt.
    """
    # check the field of the message
    field = msg['field']

    # If the message is a FitRes, save model and upload model hash to the blockchain
    if field == 'FitRes':
        # get the message params
        parameters_prime, num_examples_train, results = msg['data']

        # hash the message params which is dictionary
        model_hash = make_hash(parameters_prime)

        # save and upload the model
        upload_model(s3, parameters_prime)

        # prepare the arguments for the FitRes function
        args = [model_hash, num_examples_train]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'FitRes')(*args)

    # If the message is a EvaluateRes, upload the result to the blockchain
    elif field == 'EvaluateRes':
        # get the message params
        args = [msg['data']]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'EvaluateRes')(*args)

    # Create a transaction for calling the function
    transaction = function.buildTransaction()

    # Estimate the gas required for the transaction
    gas_estimate = w3.eth.estimateGas(transaction)

    # Set up the transaction parameters
    transaction.update({
    'from': sender_address,
        'to': contract.address,
        'gas': gas_estimate,
        'nonce': w3.eth.getTransactionCount(sender_address),
    })

    signed_transaction = w3.eth.account.signTransaction(transaction, account.key)
    transaction_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)
    transaction_receipt = w3.eth.waitForTransactionReceipt(transaction_hash)

    return transaction_receipt



def start_web3_client(client, contract_address, abi):

    while True:
        s3 = s3_connection()
        w3 = web3.Web3(web3.HTTPProvider("http://localhost:7545"))
        contract = w3.eth.contract(address=contract_address, abi=abi)
        (receive, send) = web3_connection(s3, w3, contract)

        while True:
            server_message = receive()
            print(server_message)
            client_message, sleep_duration, keep_going = handle_receive(
                client, server_message, s3, contract
            )
            send(client_message)
    
            if not keep_going:
                break

        # Check if we should disconnect and shut down
        if sleep_duration == 0:
            print("Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        print("Sleeping for {} seconds".format(sleep_duration))
        time.sleep(sleep_duration)

