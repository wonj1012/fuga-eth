from src.dataprocessing import *

trainset, valset, testset = load_dataset("data/web3mon.xlsx")
print(trainset[10])