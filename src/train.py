from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch

def train_epoch(model, dataloader, optimizer, scaler, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0
    loss_fn = CrossEntropyLoss(weight=class_weights)
    
    accumulation_steps = 2 
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 1. Run the forward pass in mixed precision (Updated API)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels)
            # Scale the loss to account for gradient accumulation
            loss = loss / accumulation_steps 
        
        # 2. Scale gradients and run backward pass
        scaler.scale(loss).backward()
        
        # 3. Step the optimizer only after 'accumulation_steps' batches
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            optimizer.zero_grad()
            
        # Re-multiply by accumulation_steps to keep your loss logging accurate
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(dataloader)