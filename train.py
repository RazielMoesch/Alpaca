import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .validate import validate
import wandb

def train(epochs, transformer, loss_fn, train_dl, optimizer=torch.optim.Adam, lr=1e-4, device=None, validate_data=False, 
          validation_dl=None, wandb_tracking: str = None, lr_scheduler=False):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transformer.to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    if wandb_tracking:
        wandb.init(project=wandb_tracking, config={"Learning_Rate": lr, "Epochs": epochs})
    
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(epochs):
        transformer.train()
        total_loss = 0
        count = 0
        batch_iterator = tqdm(train_dl, desc=f'Epoch {epoch+1}/{epochs}', leave=True, 
                             bar_format='{l_bar}{bar:20}{r_bar}', 
                             postfix={"Batch Loss": "N/A", "Avg Loss": "N/A", "LR": f"{lr:.2e}"})
        
        for tensor_tokens in batch_iterator:
            tensor_tokens = tensor_tokens.to(device)

            out = transformer(tensor_tokens[:, :-1], tensor_tokens[:, :-1])

            target = tensor_tokens[:, 1:].to(device)

            loss = loss_fn(out.view(-1, out.size(-1)), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            
            # Update progress bar with detailed information
            avg_loss = total_loss / count
            batch_iterator.set_postfix({
                "Batch Loss": f"{loss.item():.4f}", 
                "Avg Loss": f"{avg_loss:.4f}", 
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if wandb_tracking:
                wandb.log({"batch_train_loss": loss.item(), "epoch": epoch}, commit=True)
        
        epoch_loss = total_loss / len(train_dl)

        if lr_scheduler:
            scheduler.step(epoch_loss)

        if wandb_tracking:
            wandb.log({"epoch_train_loss": epoch_loss}, commit=True)
        
        print(f"Epoch: {epoch+1} | Loss: {epoch_loss:.4f}")

        if validate_data and validation_dl:
            val_loss = validate(transformer, validation_dl, loss_fn, device)
            if wandb_tracking:
                wandb.log({"epoch_validation_loss": val_loss}, commit=True)