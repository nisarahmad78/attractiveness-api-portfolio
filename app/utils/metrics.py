# utils/metrics.py
import numpy as np

def mae(preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    return float(np.mean(np.abs(preds - trues)))

def rmse(preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    return float(np.sqrt(np.mean((preds - trues) ** 2)))

def accuracy_within(preds, trues, tol=0.5):
    preds = np.array(preds)
    trues = np.array(trues)
    return float(np.mean(np.abs(preds - trues) <= tol) * 100.0)
