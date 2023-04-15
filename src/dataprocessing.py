import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class QADataset(Dataset):
    """
    Dataset class for question-answer pairs.
    """
    def __init__(self, data: List[Tuple[str, str]], tokenizer: GPT2Tokenizer, max_length: int = 512):
        """
        Args:
        data: List of tuples containing questions and answers.
        tokenizer: Tokenizer to use for encoding the input.
        max_length: Maximum sequence length to use when encoding the input.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the encoded input sequence for the given index.
        """
        question, answer = self.data[idx]
        input_text = f"{question}{self.tokenizer.eos_token}{answer}{self.tokenizer.eos_token}"
        input_ids = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return input_ids

def load_dataset(file_path: str, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads and returns the train, validation, and test datasets.
    Args:
    file_path: Path to the Excel file containing the question-answer pairs.
    test_size: The proportion of the dataset to include in the test split.
    val_size: The proportion of the dataset to include in the validation split.
    random_state: The random state used to split the dataset.
    Returns:
    A tuple of Dataset objects containing the train, validation, and test datasets.
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the training data into training and validation sets
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)

    # Create the datasets
    tokenizer = GPT2Tokenizer.from_pretrained("hivemind/gpt-j-6B-8bit")
    trainset = QADataset(train_df[['Content', 'Content']].values.tolist(), tokenizer)
    valset = QADataset(val_df[['Content', 'Content']].values.tolist(), tokenizer)
    testset = QADataset(test_df[['Content', 'Content']].values.tolist(), tokenizer)

    return trainset, valset, testset



# 데이터셋 불러오기
dataset = load_dataset('data/web3mon.xlsx')

# DataLoader로 변환
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

def add_username_prefix(username: str, data: str):
    return username + ": " + data


def preprocess_input(data: str):
    # do some preprocessing
    return data