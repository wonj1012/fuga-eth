import torch
from transformers import AutoModelForSequenceClassification
from collections import OrderedDict
import flwr as fl
from clientUtils import *

class IMDBClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        self.net = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=2
        ).to(DEVICE)
        self.trainloader, self.testloader = load_data()
        self.config = None
    
    def init_model(self):
        self.net = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=2
        ).to(DEVICE)

    def set_config(self, config) :
        self.config = config

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=self.config["local_epochs"], learning_rate=self.config["learning_rate"])
        print("Training Finished.")
        print("Test Results : ", test(self.net, self.testloader))
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}