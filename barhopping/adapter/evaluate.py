import numpy as np
import matplotlib.pyplot as plt

def evaluate(anchors: np.ndarray, positives: np.ndarray, true_ids: np.ndarray, k: int = 20):

    sims  = anchors @ positives.T
    ranks = np.argsort(-sims, axis=1)
    rr = []
    hits = []
    for i, tid in enumerate(true_ids):
        pos = np.where(ranks[i] == (tid - 1))[0]
        if len(pos):
            rr.append(1.0 / (pos[0] + 1))
            hits.append(pos[0] < k)
        else:
            rr.append(0.0)
            hits.append(False)
    return float(np.mean(rr)), float(np.mean(hits))


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Adapter Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()