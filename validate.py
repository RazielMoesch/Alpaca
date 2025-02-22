import torch

def validate(model, val_dl, loss_fn, device=None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    avg_loss = 0
    count = 0
    for batch in val_dl:
        model.eval()
        with torch.inference_mode():
            batch.to(device)
            out = model(batch, batch)
            target = batch.to(device)
            out = out.to(device)
            loss = loss_fn(out.view(-1, out.size(-1)), target.view(-1))
            avg_loss += loss
            count +=1 
    loss = avg_loss/count
    print(f'Validation Loss: {loss:.4}')
    return loss



