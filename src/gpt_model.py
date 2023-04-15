import torch
import tqdm
import transformers
from gpt8bit import *
from math import exp


def train_gpt(gpt, trainloader, valloader, epochs, lr, device: str = "cpu"):
    
    add_adapters(gpt)
    gpt.to(device)

    gpt.gradient_checkpointing_enable()
    optimizer = Adam8bit(gpt.parameters(), lr=lr, weight_decay=0.01)

    num_training_steps = epochs * len(trainloader)

    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, int(num_training_steps*0.1), num_training_steps
    )

    filepath = '/model.pt'

    scaler = torch.cuda.amp.GradScaler()

    progress_bar = tqdm(total=num_training_steps)
    k = 0

    for epoch in range(epochs):
        for batch in trainloader:

            k = k + 1
            if k % 500 == 0:
                print(k)
                state = {'k' : k, 'epoch': epochs, 'lr_scheduler': lr_scheduler.state_dict(), 'state_dict': gpt.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, filepath)
            
            # Extract question and answer tokens from the batch
            questions, answers = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = gpt(input_ids=questions)

                loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), answers[:, 1:].flatten(),
                                       reduction='mean', ignore_index=-100)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()
            progress_bar.update(1)
    
    train_loss, train_ppl = test_gpt(gpt, trainloader)
    val_loss, val_ppl = test_gpt(gpt, valloader)

    results = {
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }
    return results


def test_gpt(gpt, testloader, device: str = "cpu"):
    gpt.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in testloader:
            # Extract question and answer tokens from the batch
            questions, answers = batch["questions"].to(device), batch["answers"].to(device)

            with torch.cuda.amp.autocast():
                out = gpt(input_ids=questions)

                loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), answers[:, 1:].flatten(),
                                       reduction='sum', ignore_index=-100)

            total_loss += loss.item()
            total_tokens += answers[:, 1:].ne(-100).sum().item() # Assuming -100 is the padding token index

    avg_loss = total_loss / total_tokens
    ppl = exp(avg_loss)

    return avg_loss, ppl