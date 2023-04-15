from transformers import GPTJForCausalLM, GPT2Tokenizer
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from src.dataprocessing import *

from src.gpt_model import train_gpt

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
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.config: Dict[str, Any] = config
        self.trainset: Dataset = None
        self.valset: Dataset = None
        self.testset: Dataset = None
        # self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")
    
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
    
    def fit(self, parameters: dict) -> Tuple[dict, int, Dict[str, Any]]:
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
        result = train_gpt(self.model, trainloader, valloader, epochs=self.config["local_epochs"], lr=self.config["learning_rate"], tokenizer=self.tokenizer, device=self.device)
        return self.get_parameters(), len(trainloader), result
    
    
    def evaluate(self, parameters: dict) -> Tuple[float, float]:
        """
        Evaluate the model using the given parameters and return the loss and accuracy.

        Args:
            parameters (dict): The dictionary containing model parameters to use for evaluation.
            config (dict): The dictionary containing configuration parameters for evaluation.

        Returns:
            Tuple[float, float]: A tuple containing the loss and accuracy.
        """
        self.set_parameters(parameters)
        testloader = DataLoader(self.testset, batch_size=self.config["batch_size"], shuffle=True)
        loss, accuracy = evaluate_gpt(self.model, self.tokenizer, testloader)
        return loss, accuracy
    
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
