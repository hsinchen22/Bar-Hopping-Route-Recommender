import numpy as np
import sqlite3, ast
from barhopping.embedding.granite import get_embedding
from .reranker import get_reranker
from .config import BARS_DB
from barhopping.logger import logger
from typing import List, Dict, Union

class VectorSearch:
    def __init__(self, db_path: str = BARS_DB):
        """Initialize the vector search.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load all embeddings from the database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                'SELECT id, name, summary, embedding, address, photo FROM bars'
            ).fetchall()
            
        self.ids, self.names, self.summaries, self.embeddings, self.addresses, self.img_urls = [], [], [], [], [], []
        for i, n, s, e, a, img in rows:
            self.ids.append(i)
            self.names.append(n)
            self.summaries.append(s)
            self.embeddings.append(np.array(ast.literal_eval(e), dtype=np.float32))
            self.addresses.append(a)
            self.img_urls.append(img)
            
        if self.embeddings:
            self.embeddings = np.vstack(self.embeddings)
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
        else:
            logger.warning("No embeddings found in database")
            self.embeddings = np.array([])
            
    def search(self, query: str, top_k: int = 10, rerank: bool = True, threshold: float = None) -> List[Dict[str, Union[str, float]]]:
        """Search for similar bars using vector search and optional reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to apply reranking (default: True)
            threshold: Reranking score threshold (optional)
        Returns:
            List of dictionaries containing bar information and scores
        """
        if len(self.embeddings) == 0:
            logger.error("No embeddings available for search")
            return []
            
        # Get query embedding
        qv = get_embedding(query).cpu().numpy().reshape(-1)
        
        # Calculate similarities
        sims = self.embeddings @ qv
        idx = np.argsort(sims)[-top_k:][::-1]
        
        # Prepare candidates for reranking
        candidates = [
            {
                "id": self.ids[i],
                "name": self.names[i],
                "summary": self.summaries[i],
                "address": self.addresses[i],
                "photo": self.img_urls[i],
                "vector_score": float(sims[i])
            }
            for i in idx
        ]
        
        if rerank:
            logger.info("Applying reranking...")
            reranker = get_reranker()
            reranked = reranker.rerank(query, candidates, top_k=top_k, threshold=threshold)
            return reranked
        else:
            return candidates
            
    def refresh(self):
        """Reload embeddings from the database."""
        self._load_embeddings()
        
# Global vector search instance
_vector_search = None

def get_vector_search() -> VectorSearch:
    """Get the global vector search instance."""
    global _vector_search
    if _vector_search is None:
        _vector_search = VectorSearch()
    return _vector_search
