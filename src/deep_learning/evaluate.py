"""Evaluation utilities for the CNN."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

@torch.no_grad()
def predict(model, loader, device="cuda"):
    device=torch.device(device if torch.cuda.is_available() else "cpu"); model.eval().to(device)
    y_true=[]; y_pred=[]
    for x,y in loader:
        pred=model(x.to(device)).argmax(1).cpu().numpy(); y_pred.extend(pred.tolist()); y_true.extend(y.numpy().tolist())
    return np.array(y_true),np.array(y_pred)

def classification_metrics(y_true,y_pred):
    return {"accuracy":accuracy_score(y_true,y_pred),"macro_f1":f1_score(y_true,y_pred,average="macro"),"confusion_matrix":confusion_matrix(y_true,y_pred),"report":classification_report(y_true,y_pred,digits=4)}
