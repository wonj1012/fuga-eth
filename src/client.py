from transformers import GPTJForCausalLM, GPT2Tokenizer
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

import utils

class GPTClient:
    def __init__(self, config: dict) -> None:
        """
        Constructor of GPTClient class.

        Args:
            config (dict): The dictionary containing configuration parameters for GPTClient.

        Returns:
            None
        """
        self.model: GPTJForCausalLM = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("hivemind/gpt-j-6B-8bit")
        self.config: dict = config
        self.trainset: Dataset = None
        self.valset: Dataset = None
        self.testset: Dataset = None
    
    def load_dataset(self, trainset: Dataset, valset: Dataset, testset: Dataset) -> None:
        """
        Load datasets for training.

        Args:
            trainset (Dataset): The training dataset.
            valset (Dataset): The validation dataset.
            testset (Dataset): The test dataset.

        Returns:
            None
        """
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        
    def get_parameters(self) -> dict:
        """
        Return the trained model parameters as a dictionary.

        Returns:
            dict: The dictionary containing trained model parameters.
        """
        return self.model.state_dict()
    
    def set_parameters(self, parameters: dict) -> None:
        """
        Set model parameters using a dictionary of parameters.

        Args:
            parameters (dict): The dictionary containing the model parameters to set.

        Returns:
            None
        """
        self.model.load_state_dict(parameters)
    
    def fit(self, parameters: dict, config: dict) -> Tuple[dict, int]:
        """
        Train the model and return the trained parameters and number of batches.

        Args:
            parameters (dict): The dictionary containing model parameters to use for training.
            config (dict): The dictionary containing configuration parameters for training.

        Returns:
            Tuple[dict, int]: A tuple containing trained parameters dictionary and number of batches.
        """
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
    
    def generate_answer(self, data: str, config: dict) -> str:
        """
        Generate an answer to the given data.

        Args:
            data (str): The input data to generate an answer for.
            config (Dict): The dictionary containing configuration parameters for model generation.

        Returns:
            str: The generated answer for the given input data.
        """
        data = utils.preprocess_input(data)
        input_ids = self.tokenizer.encode(data, return_tensors="pt")
        generated_tokens = self.model.generate(input_ids, **config)
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return answer
