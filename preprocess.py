import pandas as pd
import re
from typing import Dict
import nltk
from nltk.corpus import stopwords
import json


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
    for username in special_username_dict:
        df.loc[df['Author username'] == username, 'Content'] = special_username_dict[username] + ': ' + df['Content'].astype(str)
    return df


def nltk_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # Clean the text data
    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
        text = re.sub(r'\d+', '', text)      # Remove numbers
        text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
        return text.strip()

    df['Content'] = df['Content'].apply(clean_text)

    # Normalize the text data
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    # Lowercase, remove stop words
    df['Content'] = df['Content'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in stop_words]))
    df = df[df['Content'] != '']
    
    return df

def remove_links_and_emails(df: pd.DataFrame) -> pd.DataFrame:
    def clean_links_and_emails(text):
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove other links (e.g., ftp, file, etc.)
        text = re.sub(r'\b(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+:\\/\\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        return text

    df['Content'] = df['Content'].apply(clean_links_and_emails)
    
    return df



df = pd.read_excel('data/web3mon.xlsx')[['Author username', 'Content']]

special_user = {'Jaewon': 'gm'}
df = df.pipe(groupby_username).pipe(remove_links_and_emails).pipe(nltk_preprocessing)
df = df.pipe(nltk_preprocessing).pipe(add_prefix_special_username, special_username_dict=special_user)

training_data = []
for i in range(len(df['Content'])-1):
    training_data.append({'prompt': df['Content'][i] + "\n\n###\n\n", 'completion': df['Content'][i+1]})
    
json.dump(training_data, open('data/web3mon.jsonl', 'w'), indent=4)