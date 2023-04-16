import argparse
import json
from typing import Callable, Dict, Tuple
import time
import web3
from web3_client_utils import *
from imdbClient import IMDBClient

PUBLIC_KEY = os.environ.get('PUBLIC_KEY')
PRIVATE_KEY = os.environ.get('PRIVATE_KEY')

def listen_for_event(contract, event_name):
    # create a filter to listen for the specified event
    event_filter = contract.events[event_name].createFilter(fromBlock='latest')

    while True:
        # check if any new events have been emitted
        for event in event_filter.get_new_entries():
            try:
                # if the specified event has been emitted, return its message
                if event.event == event_name:
                    yield event.args
            except Exception as e:
                print(f"Error processing event: {e}")
        # wait for new events
        time.sleep(5)

def web3_connection(s3, w3, contract) -> Tuple[Callable[[], Dict], Callable[[Dict], None]]:
    """
    Creates a connection to the blockchain and returns a function to receive messages and a function to send messages.
    """
    web3_message_iterator = listen_for_event(contract, "ServerMessage")

    def receive():
        try:
            return next(web3_message_iterator)
        except StopIteration:
            return None

    def send(msg):
        return handle_send(msg, s3, w3, contract)

    return (receive, send)


def handle_receive(client, msg, s3, w3, contract):
    """
    Handles a received message.
    Returns a tuple containing the response message, the number of samples and a boolean indicating whether the client should shut down.
    """
    # check the field of the message
    field = msg['field']

    if field == "Finished":
        print("finished")
        return None, 0, False

    # if the message is a request for Ready, join the round
    if field == "Ready":
        print("join round")
        function = getattr(contract.functions, 'joinRound')()

    # if the message is a request for JoinRound, after few minutes, start the round
    if field == "JoinRound":
        time.sleep(1*20)
        print("call start round")
        function = getattr(contract.functions, 'startRound')()

    if field == "Ready" or field == "JoinRound":
        send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)
        return None, 0, True

    # if the message is a request for the ConfigIns, set the Config data
    if field == "ConfigIns":
        print("config start")
        # receive the Config data
        response = read_transaction(w3, contract, 'getConfig', PUBLIC_KEY, PRIVATE_KEY)
        
        self_centered = response['self_centered']
        batch_size = response['batch_size']
        learning_rate = float(response['learning_rate'])
        local_epochs = response['local_epochs']
        val_steps = response['val_steps']

        config = {'self_centered': self_centered, 'batch_size': batch_size, 'learning_rate' : learning_rate ,'local_epochs': local_epochs, 'val_steps': val_steps}

        client.set_config(config)

        return {'field':'ConfigRes'}, 0, True
        
    # if the message is a request for the FitIns, aggregate the model and return the result after fit
    if field == "FitIns":
        print("fit start")
        # receive the Client data
        response = read_transaction(w3, contract, 'getClient', PUBLIC_KEY, PRIVATE_KEY)
        print(response)

        self_model_hash = response['model_hash']
        self_num_sample = response['num_sample']
        self_score = response['score']

        model_hashes = []
        num_samples = []
        scores = []

        response = read_transaction(w3, contract, 'FitIns', PUBLIC_KEY, PRIVATE_KEY)
        print(response)

        other_model_hashes = response['model_hashes']
        other_num_samples =response['num_samples']
        other_scores = response['scores']

        # if there is no other model, fit the model on client
        if(other_model_hashes == ['']):
            if(self_model_hash==''):
                fitres = client.fit(client.get_parameters())
        # if there is other model, aggregate the model
        else:
            for model_hash in other_model_hashes:
                object_key = f'models/{model_hash}.bin'
                file_path = object_key
                s3.download_file(BUCKET_NAME, object_key, file_path)

                # check hash value
                if(check_model(model_hash)):
                    model_hashes.append(model_hash)
                    num_samples.append(other_num_samples)
                    scores.append(other_scores)

            if(check_model(self_model_hash)):
                model_hashes.append(self_model_hash)
                num_samples.append(self_num_sample)
                scores.append(sum(scores) if client.self_centered else self_score)
                if(len(scores)==1, scores[0]==0):
                    scores[0] = 1

            # aggregate the model
            fitres = aggregate_fit(client, model_hashes, num_samples, scores)

        # return the result of fit
        return {'field':'FitRes', 'data': fitres},0, True
    
    # if the message is a request for the EvaluateIns, evaluate the model and return the result
    elif field == "EvaluateIns":
        print("eval start")
        # evaluate the model on client
        response = read_transaction(w3, contract, 'EvaluateIns', PUBLIC_KEY, PRIVATE_KEY)
        print(response)
        model_hashes = response['model_hashes']
        evalres = {}

        for model_hash in model_hashes:
            object_key = f'models/{model_hash}.bin'
            file_path = object_key

            s3.download_file(BUCKET_NAME, object_key, file_path)

            # check hash value
            check_model(model_hash)
            param = read_model(model_hash)

            # evaluate the model
            loss, _, _ = client.evaluate(param)
            evalres[model_hash] = loss

        return {'field':'EvaluateRes', 'data':evalres}, 0 , True


def handle_send(msg, s3, w3, contract):
    """
    Handles a message to be sent.
    Returns the transaction receipt.
    """
    # check the field of the message
    field = msg['field']

    # If the message is a ConfigRes, upload the result to the blockchain
    if field == 'ConfigRes':
        print("config complete")
        function = getattr(contract.functions, 'ConfigRes')()

    # If the message is a FitRes, save model and upload model hash to the blockchain
    if field == 'FitRes':
        print("fit complete")
        # get the message params
        parameters_prime, num_examples_train, results = msg['data']

        # hash the message params which is dictionary
        model_hash = make_hash(parameters_prime)

        # save and upload the model
        upload_model(s3, parameters_prime,model_hash)

        # prepare the arguments for the FitRes function
        args = [model_hash, num_examples_train]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'FitRes')(*args)

    # If the message is a EvaluateRes, upload the result to the blockchain
    elif field == 'EvaluateRes':
        print("eval complete")
        # get the message params
        evalres = msg['data']
        model_hashes = list(evalres.keys())
        values = [int(value) for value in evalres.values()]
        args = [model_hashes, values]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'EvaluateRes')(*args)

    send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)



def start_web3_client(client, contract_address, abi):

    while True:
        s3 = s3_connection()
        w3 = web3.Web3(web3.HTTPProvider(HTTP_PROVIDER))
        contract = w3.eth.contract(address=contract_address, abi=abi)
        receive,send = web3_connection(s3, w3, contract)
        # receive,send = next(conn)

        while True:
            server_message = receive()
            print("server message : ",server_message)
            if(server_message is not None):
                if(server_message['field'] == "JoinRound" and server_message['sender'] != PUBLIC_KEY):
                    continue
                client_message, sleep_duration, keep_going = handle_receive(
                    client, server_message, s3, w3, contract
                )
                if(client_message is not None):
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

if __name__ == "__main__":
    # get the contract address and abi from the config file'
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c','--contract_address', required=True, type=str, default='config.json')
    parser.add_argument('-p','--public_key', type=str, default=PUBLIC_KEY)
    parser.add_argument('-m','--private_key', type=str, default=PRIVATE_KEY)
    args = parser.parse_args()

    # contract_address = args.contract_address
    contract_address = input("contract address : ")
    PUBLIC_KEY = args.public_key
    PRIVATE_KEY = args.private_key
    print(PUBLIC_KEY, PRIVATE_KEY)
    abi = json.load(open('contract/build/contracts/FugaController.json', 'r'))['abi']

    # get the client
    client = IMDBClient()

    # start the client
    start_web3_client(client, contract_address, abi)