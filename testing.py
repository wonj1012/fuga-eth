from src.dataprocessing import *

trainset, valset, testset = load_dataset("data/web3mon.xlsx")
for i in range(len(trainset)):
    if trainset[i][0][-1] != 50400:
        print(trainset[i])
