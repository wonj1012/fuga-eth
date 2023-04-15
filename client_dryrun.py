from src.client import GPTClient

client = GPTClient({"batch_size": 1, "learning_rate": 1e-4, "local_epochs": 1, "val_steps": 1})
client.load_dataset("data/web3mon.xlsx")
client.fit(client.get_parameters())