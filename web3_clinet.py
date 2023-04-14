from typing import Callable, Dict, Tuple
from dotenv import load_dotenv
import boto3
import os 
import time


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

def web3_connection(contract) -> Tuple[Callable[[], Dict]]:
    web3_message_iterator = listen_for_event(contract, "ServerMessage")

    receive: Callable[[], Dict] = lambda: next(web3_message_iterator)

    return receive