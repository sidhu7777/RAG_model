import json
import faiss
from sentence_transformers import SentenceTransformer
from src.components.retriever import Retriever
from src.logger import logging
from src.exception import CustomException

def test_retriever():
    # Paths to necessary files
    faiss_index_path = "C:/Users/91832/Desktop/RAG_Model_development/vector_db/vector_index.faiss"
    mapping_path = "C:/Users/91832/Desktop/RAG_Model_development/vector_db/enhanced_unified_mapping.json"

    try:
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        logging.info(f"FAISS index loaded from {faiss_index_path}")

        # Load unified mapping
        with open(mapping_path, "r") as file:
            unified_mapping = json.load(file)
        logging.info(f"Unified mapping loaded from {mapping_path}")

        # Initialize SentenceTransformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model initialized.")

        # Create Retriever instance
        retriever = Retriever(
            model=model,
            index=index,
            mapping_data=unified_mapping,
            similarity_threshold=0.5,
        )

        # Test with a sample query
        query = "What is revenue?"
        results = retriever.retrieve(query, k=5)

        # Assert and log the results
        assert results, "No results returned for the query."
        for i, (content, distance) in enumerate(results):
            print(f"Result {i+1}: {content}")
            print(f"Distance: {distance}")

    except Exception as e:
        logging.error("Error testing the Retriever.")
        raise CustomException("Critical failure during Retriever testing.", e)

if __name__ == "__main__":
    test_retriever()
