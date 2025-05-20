import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
from barhopping.logger import logger

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        """Initialize the reranker model.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on (cpu, cuda, mps). If None, uses the default from config.
        """
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else 
                               "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading reranker model {self.model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        
    def rerank(self, query: str, candidates: List[Dict[str, str]], top_k: int = 5, threshold: float = None) -> List[Dict[str, Union[str, float]]]:
        """Rerank candidates based on their relevance to the query.
        
        Args:
            query: The search query
            candidates: List of candidate dictionaries with 'tag_name' and 'summary'
            top_k: Number of top results to return (default: 5)
            threshold: Minimum score threshold (optional)
        Returns:
            List of reranked candidates with scores
        """
        if not candidates:
            return []
        
        # Prepare input pairs
        pairs = [(query, f"{candidate["name"]}: {candidate["summary"]}") for candidate in candidates]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()
        
        # Convert to list and attach to candidates
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy().tolist()
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)
        
        # Sort by rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        if threshold is not None:
            filtered = [c for c in candidates if c["rerank_score"] >= threshold]
            return filtered
        else:
            return candidates[:top_k]

    def rerank_summary(self, query: str, ids: List[int], summaries: List[str], top_k: int = 5) -> List[int]:
        """Rerank summaries based on their relevance to the query.
        
        Args:
            query: The search query
            ids: List of IDs corresponding to the summaries
            summaries: List of summaries to rerank
            top_k: Number of top results to return
            
        Returns:
            List of IDs of the top-k reranked summaries
        """
        if not summaries:
            return []
            
        # Prepare input pairs
        pairs = [[query, summary] for summary in summaries]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1,).float()
            
        # Get top-k indices
        selected = np.argsort(scores.cpu().numpy())[-top_k:][::-1]
        
        # Return corresponding IDs
        return [ids[i] for i in selected]

# Global reranker instance
_reranker = None

def get_reranker() -> Reranker:
    """Get the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
