from transformers import GPTJForCausalLM
import torch
import utils

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPTJForCausalLM, GPT2Tokenizer
import utils

class GPTClient():
    def __init__(self, config: dict):
        
        self.model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("hivemind/gpt-j-6B-8bit")
        self.config = config
        self.trainset: Dataset = None
        self.valset: Dataset = None
        self.testset: Dataset = None
    
    def load_dataset(self, trainset, valset, testset):
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        
    def get_parameters(self):
        return self.model.state_dict()
    
    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainloader = DataLoader(self.trainset, batch_size=config["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            input_ids, labels = batch
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        return self.get_parameters(), len(trainloader)
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        valloader = DataLoader(self.valset, batch_size=config["batch_size"], shuffle=False)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in valloader:
                input_ids, labels = batch
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        return total_loss / len(valloader)
    
    def generate_answer(self, data: str, config: dict) -> str:
        data = utils.preprocess_input(data)
        input_ids = self.tokenizer.encode(data, return_tensors="pt")
        generated_tokens = self.model.generate(input_ids, **config)
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return answer

    
    
    