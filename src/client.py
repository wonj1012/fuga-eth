from transformers import GPTJForCausalLM, GPT2Tokenizer
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataprocessing import *

from gpt_model import train_gpt

class GPTClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Constructor of GPTClient class.

        Args:
            config (dict): {batch_size (int): The batch size to use for training,
                            learning_rate (float): The learning rate to use for training,
                            local_epochs (int): The number of epochs to train for,
                            val_steps (int): The number of steps to validate for}

        Returns:
            None
        """
        self.model: GPTJForCausalLM = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("hivemind/gpt-j-6B-8bit")
        self.config: Dict[str, Any] = config
        self.trainset, self.valset, self.testset: Dataset = None, None, None
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration parameters.

        Args:
            config (dict): The dictionary containing configuration parameters for GPTClient.

        Returns:
            None
        """
        self.config = config
    
    def load_dataset(self, path: str = "data/web3mon.xlsx") -> None:
        """
        Load datasets for training.

        Args:
            trainset (Dataset): The training dataset.
            valset (Dataset): The validation dataset.
            testset (Dataset): The test dataset.

        Returns:
            None
        """
        self.trainset, self.valset, self.testset = get_dataset(path)
        
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
    
    def fit(self, parameters: dict) -> Tuple[dict, int]:
        """
        Train the model and return the trained parameters and number of batches.

        Args:
            parameters (dict): The dictionary containing model parameters to use for training.
            config (dict): The dictionary containing configuration parameters for training.

        Returns:
            Tuple[dict, int]: A tuple containing trained parameters dictionary and number of batches.
        """        
        self.set_parameters(parameters)
        trainloader = DataLoader(self.trainset, batch_size=self.config["batch_size"], shuffle=True)
        valloader = DataLoader(self.valset, batch_size=self.config["batch_size"], shuffle=True)
        result = train_gpt(self.model, self.tokenizer, trainloader, valloader, self.config["learning_rate"], self.config["local_epochs"], self.config["val_steps"])
        return self.get_parameters(), len(trainloader), result
    
    
    def generate_answer(self, data: str, config: dict) -> str:
        """
        Generate an answer to the given data.

        Args:
            data (str): The input data to generate an answer for.
            config (Dict): The dictionary containing configuration parameters for model generation.

        Returns:
            str: The generated answer for the given input data.
        """
        data = preprocess_input(data)
        input_ids = self.tokenizer.encode(data, return_tensors="pt")
        generated_tokens = self.model.generate(input_ids, **config)
        answer = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return answer
