from src.client import GPTClient

client = GPTClient({"batch_size": 1, "learning_rate": 1e-4, "local_epochs": 1, "val_steps": 1})
client.load_dataset("data/web3mon.xlsx")
print(client.generate_answer("What is the name of the first Pokemon?", {"max_length": 10, "num_beams": 5, "temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.2, "do_sample": True, "early_stopping": True, "no_repeat_ngram_size": 2, "num_return_sequences": 3}))
client.fit(client.get_parameters())