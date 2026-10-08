"""Training utilities for the baseline CNN."""
from pathlib import Path
import torch
from torch import nn

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); loss_sum=0.0; correct=0; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
        logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
        loss_sum += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum=0.0; correct=0; total=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x); loss=criterion(logits,y)
        loss_sum += loss.item()*x.size(0); correct += (logits.argmax(1)==y).sum().item(); total += y.size(0)
    return loss_sum/total, correct/total

def fit(model, train_loader, val_loader, epochs=20, learning_rate=1e-3, device="cuda", checkpoint_path=None):
    device=torch.device(device if torch.cuda.is_available() else "cpu"); model.to(device)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    history=[]; best_val=-1.0
    for epoch in range(1,epochs+1):
        tr_loss,tr_acc=train_one_epoch(model,train_loader,optimizer,criterion,device)
        va_loss,va_acc=evaluate(model,val_loader,criterion,device)
        history.append({"epoch":epoch,"train_loss":tr_loss,"train_accuracy":tr_acc,"val_loss":va_loss,"val_accuracy":va_acc})
        print(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} | train acc {tr_acc:.4f} | val loss {va_loss:.4f} | val acc {va_acc:.4f}")
        if checkpoint_path and va_acc>best_val:
            best_val=va_acc; Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state_dict":model.state_dict(),"val_accuracy":va_acc,"epoch":epoch},checkpoint_path)
    return history
