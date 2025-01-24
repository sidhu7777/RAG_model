from sentence_transformers import SentenceTransformer
import faiss
import json
from src.logger import logging
from src.exception import CustomException


class Retriever:
    def __init__(self, model, index, mapping_data, similarity_threshold=1.5):
        """
        Initialize the retriever.
        Args:
            model: Embedding model (e.g., SentenceTransformer).
            index: FAISS index.
            mapping_data: Unified mapping with text and table content.
            similarity_threshold: Minimum similarity score to include a result.
        """
        try:
            self.model = model
            self.index = index
            self.mapping_data = mapping_data
            self.similarity_threshold = similarity_threshold
            logging.info("Retriever initialized successfully.")
        except Exception as e:
            logging.error("Error initializing Retriever: %s", str(e))
            raise CustomException("Failed to initialize Retriever.", e)

    def retrieve(self, query, k=5):
        """
        Retrieve the top-k most relevant chunks for a given query.
        Args:
            query (str): User query in natural language.
            k (int): Number of top results to retrieve.
        Returns:
            List of tuples: [(content, distance), ...]
        """
        try:
            logging.info(f"Retrieving results for query: {query}")

            # Generate embedding for the query
            query_embedding = self.model.encode(query)
            logging.info("Query embedding generated successfully.")

            # Search FAISS index
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            logging.info(f"FAISS search completed. Retrieved {len(indices[0])} results.")
            logging.info(f"Distances retrieved: {distances[0]}")
            logging.info(f"Indices retrieved: {indices[0]}")

            # Map indices to human-readable results
            results = []
            for idx, i in enumerate(indices[0]):
                if distances[0][idx] < self.similarity_threshold:  # Apply similarity threshold
                    if 0 <= i < len(self.mapping_data):  # Ensure index is valid
                        item = self.mapping_data[i]
                        if "chunk_index" in item:
                            # Handle text entries
                            results.append((f"Text: {item.get('content', 'No content available')}", distances[0][idx]))
                        elif "file_name" in item and "row_index" in item:
                            # Handle table entries
                            table_info = f"Table: {item['file_name']}, Row {item['row_index']}"
                            row_content = f"Content: {item.get('row_content', 'No content available')}"
                            results.append((f"{table_info}\n{row_content}", distances[0][idx]))
                        else:
                            logging.warning(f"Unrecognized mapping format for index {i}. Skipping.")
                    else:
                        logging.warning(f"Invalid index {i} retrieved from FAISS for query: {query}")
                else:
                    logging.warning(f"Low similarity score for index {i}. Skipping.")
            return results
        except Exception as e:
            logging.error("Error during retrieval process: %s", str(e))
            raise CustomException("Failed to retrieve results.", e)


if __name__ == "__main__":
    try:
        # Load FAISS index
        faiss_index_path = "C:/Users/91832/Desktop/RAG_Model_development/vector_db/vector_index.faiss"
        index = faiss.read_index(faiss_index_path)
        logging.info(f"FAISS index loaded successfully from {faiss_index_path}")

        # Load mapping data
        mapping_path = "C:/Users/91832/Desktop/RAG_Model_development/vector_db/vector_db.index"
        with open(mapping_path, "r", encoding="utf-8") as file:
            unified_mapping = json.load(file)
        logging.info(f"Unified mapping loaded successfully from {mapping_path}")

        # Initialize SentenceTransformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model initialized.")

        # Create Retriever instance
        retriever = Retriever(
            model=model,
            index=index,
            mapping_data=unified_mapping,
            similarity_threshold=1.5,  # Adjusted similarity threshold
        )

        # Example query
        query = "What is revenue?"
        results = retriever.retrieve(query, k=5)

        # Print results
        if results:
            for i, (content, distance) in enumerate(results):
                print(f"Result {i+1}:\n{content}\nDistance: {distance}\n")
        else:
            logging.info("No results found for the query.")

    except Exception as e:
        logging.error("Critical error in Retriever setup: %s", str(e))
        raise CustomException("Critical error in Retriever setup.", e)
