import torch, os
from torch import nn
from transformers import AutoTokenizer, AutoModel
from barhopping.config import GRANITE_MODEL

tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL)
model_em = AutoModel.from_pretrained(GRANITE_MODEL).eval()

# Adapter setup
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "../../adapter/adapter_model.pth")
ADAPTER_PATH = os.path.abspath(ADAPTER_PATH)

class LinearAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x):
        return self.linear(x)

adapter = None
# Try to load adapter if available
with torch.no_grad():
    try:
        # Get hidden size from model config
        hidden_size = model_em.config.hidden_size
        adapter = LinearAdapter(hidden_size)
        if os.path.exists(ADAPTER_PATH):
            adapter.load_state_dict(torch.load(ADAPTER_PATH, map_location="cpu"))
            adapter.eval()
        else:
            adapter = None
    except Exception as e:
        print(f"[Warning] Could not load adapter: {e}")
        adapter = None

def get_embedding(text: str) -> torch.Tensor:
    inp = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model_em(**inp)[0][:, 0]
        if adapter is not None:
            out = adapter(out)
    return torch.nn.functional.normalize(out, dim=1)
