from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pandas as pd

class QADataset(Dataset):
    def __init__(self, questions: List[str], answers: List[str], tokenizer: GPT2Tokenizer):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        max_length = 512
        input_text = question + self.tokenizer.eos_token + answer + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

        return input_ids

def load_dataset(file_path: str):
    # 파일로부터 질문과 답변 쌍을 읽어옴
    # xlsx 파일을 pandas DataFrame으로 읽기
    df = pd.read_excel(file_path)

    # Content column을 기준으로 질문과 답변을 나누어 리스트에 저장
    questions = []
    answers = []
    for content in df['Content']:
        qa = content.split('\n')
        questions.append(qa[0])
        answers.append(qa[1])

    # 질문과 답변 쌍을 구분하고 리스트에 저장
    questions = []
    answers = []
    for line in lines:
        question, answer = line.strip().split('\t')
        questions.append(question)
        answers.append(answer)

    # 토크나이저 초기화
    tokenizer = GPT2Tokenizer.from_pretrained("hivemind/gpt-j-6B-8bit")

    # 토큰화된 데이터셋 생성
    dataset = QADataset(questions, answers, tokenizer)

    return dataset

# 데이터셋 불러오기
dataset = load_dataset('data/web3mon.xlsx')

# DataLoader로 변환
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

def add_username_prefix(username: str, data: str):
    return username + ": " + data


def preprocess_input(data: str):
    # do some preprocessing
    return data