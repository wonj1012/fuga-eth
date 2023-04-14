from transformers import GPTJForCausalLM
import torch
import utils


class GPTClient():
    def __init__(self, config: dict):
        self.model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        self.config = config
        self.trainset = None
        self.testset = None
        self.testset = None
        
    def get_parameters(self):
        return self.model.state_dict()
    
    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # do some training
        return self.get_parameters(), len(trainloader)
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # do some evaluation
        return 
    
    def generate_answer(self, data: str, config: dict) -> str:
        data = utils.preprocess_input(data)
        answer = self.model.generate(data)
        return answer
    
    
    