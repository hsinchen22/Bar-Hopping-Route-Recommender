import sys
from barhopping.retriever.vector_search import get_vector_search
from barhopping.embedding.granite import get_embedding
from barhopping.logger import logger
import json

def format_result(result: dict, index: int) -> str:
    """Format a single search result for display."""
    # Get values with defaults for missing fields
    name = result.get('tag_name', 'Unknown')
    summary = result.get('summary', 'No summary available')
    vector_score = result.get('vector_score', 0.0)
    rerank_score = result.get('rerank_score', 0.0)
    
    return f"""
Result {index + 1}:
Name: {name}
Summary: {summary}
Vector Score: {vector_score:.3f}
Rerank Score: {rerank_score:.3f}
"""

def main():
    # Initialize vector search
    vector_search = get_vector_search()
    
    print("\nBar Search (type 'quit' to exit)")
    print("Enter your search query:")
    
    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() == 'quit':
                break
                
            if not query:
                continue
                
            # Perform search
            results = vector_search.search(query, top_k=5)
            
            if not results:
                print("No results found.")
                continue
                
            # Display results
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results):
                print(format_result(result, i))
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 