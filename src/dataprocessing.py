import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

class QADataset(Dataset):
    """
    Dataset class for question-answer pairs.
    """
    def __init__(self, data: List[Tuple[str, str]], tokenizer: GPT2Tokenizer, max_length: int = 128):
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
    df = pd.read_excel(file_path)[['Author username', 'Content']]
    
    special_user = {'Jaewon': 'admin'}
    df = df.pipe(groupby_username).pipe(add_prefix_special_username, special_username_dict=special_user)

    # Split the data into training and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the training data into training and validation sets
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)

    # Create the datasets
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    trainset = QADataset(train_df[['Content', 'Content']].values.tolist(), tokenizer)
    valset = QADataset(val_df[['Content', 'Content']].values.tolist(), tokenizer)
    testset = QADataset(test_df[['Content', 'Content']].values.tolist(), tokenizer)

    return trainset, valset, testset


def groupby_username(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby((df['Author username'] != df['Author username'].shift()).cumsum()).apply(lambda x: pd.Series({'Author username': x['Author username'].iloc[0], 'Content': ' '.join(x['Content'].fillna(''))})).reset_index(drop=True)
    return df

def normalize_content(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    
    return df


def add_prefix_special_username(df: pd.DataFrame, special_username_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Add username prefix to special user such as admin or bot.

    Args:
        username (str): The username to match against.
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    for i in range(len(df)):
        print(df['Content'][i])
    for username in special_username_dict:
        df.loc[df['Author username'] == username, 'Content'] = special_username_dict[username] + ': ' + df['Content'].astype(str)
    return df


def nltk_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Download necessary resources from nltk
    nltk.download('stopwords')
    
    # Tokenize and clean the text data
    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
        text = re.sub(r'\d+', '', text)      # Remove numbers
        text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
        return text.strip()

    df['clean_content'] = df['Content'].apply(clean_text)

    # Normalize the text data
    stop_words = set(stopwords.words('english'))
    
    # lowercase, remove stop words
    df['clean_content'] = df['clean_content'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in stop_words]))


def remove_links_and_emails(text):
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove other links (e.g., ftp, file, etc.)
    text = re.sub(r'\b(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+:\\/\\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    return text


def preprocess_input(data: str):
    # do some preprocessing
    return data
