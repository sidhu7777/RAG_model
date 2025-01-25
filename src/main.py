import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from src.components.data_extraction import DataExtraction
from src.components.data_preprocessing import data_preprocessing_pipeline
from src.components.new_vector import VectorDBHandler
from src.components.retriever import Retriever
from src.components.response_generator import ResponseGenerator
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("Starting RAG pipeline...")

        # Step 1: Data Extraction
        logging.info("Step 1: Extracting raw data from PDFs.")
        file_path = r"C:\Users\91832\Desktop\RAG_Model_development\notebook\Sample Financial Statement.pdf"
        data_extractor = DataExtraction(file_path)
        text_path, table_dir = data_extractor.run()

        # Step 2: Data Preprocessing
        logging.info("Step 2: Preprocessing raw data.")
        data_preprocessing_pipeline()

        # Step 3: Embedding and Vector Database Creation
        logging.info("Step 3: Generating embeddings and creating vector database.")
        vector_db_handler = VectorDBHandler()

        # Load cleaned text chunks
        text_chunks_path = os.path.join("data", "processed", "cleaned_text_chunks.json")
        with open(text_chunks_path, "r", encoding="utf-8") as f:
            cleaned_text_chunks = json.load(f)

        # Generate text embeddings
        logging.info("Debugging text_embeddings in VectorDBHandler before generating embeddings...")
        text_mapping, text_embeddings = vector_db_handler.generate_text_embeddings(cleaned_text_chunks)

        # Debugging logs for generated embeddings
        logging.info("Type of text_mapping after generate_text_embeddings: %s", type(text_mapping))
        logging.info("Type of text_embeddings after generate_text_embeddings: %s", type(text_embeddings))
        if isinstance(text_mapping, list):
            logging.info("First two entries of text_mapping: %s", text_mapping[:2])
        else:
            logging.error("text_mapping is not a list. Received type: %s", type(text_mapping))
            raise ValueError("text_mapping must be a list.")

        # Ensure table directory exists and is valid
        flattened_table_folder = os.path.join("data", "processed", "flattened_table")
        if not os.path.exists(flattened_table_folder):
            raise CustomException(f"Flattened table folder not found: {flattened_table_folder}")

        # Generate table embeddings
        table_mapping, table_embeddings = vector_db_handler.generate_table_embeddings(flattened_table_folder)

        # Validate mappings
        if not isinstance(table_mapping, list):
            logging.error("table_mapping is not a list. Received type: %s", type(table_mapping))
            raise ValueError("table_mapping must be a list.")

        if not all(isinstance(entry, dict) for entry in table_mapping):
            logging.error("table_mapping contains non-dictionary entries.")
            raise ValueError("table_mapping must only contain dictionaries.")

        # Combine mappings
        unified_mapping = text_mapping + table_mapping

        # Save unified mapping
        vector_db_handler.save_unified_mapping(unified_mapping)

        # Create FAISS index
        vector_db_handler.create_faiss_index(unified_mapping, text_embeddings, table_embeddings)

        # Step 4: Retrieval and Response Generation
        logging.info("Step 4: Initializing retriever and response generator.")
        faiss_index_path = os.path.join(vector_db_handler.vector_db_path, "vector_index.faiss")
        mapping_path = os.path.join(vector_db_handler.vector_db_path, "vector_db.index")

        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        logging.info("FAISS index loaded successfully.")

        # Load unified mapping
        with open(mapping_path, "r", encoding="utf-8") as f:
            unified_mapping = json.load(f)

        # Initialize Retriever
        model = SentenceTransformer("all-MiniLM-L6-v2")
        retriever = Retriever(model=model, index=index, mapping_data=unified_mapping, similarity_threshold=1.5)

        # Initialize ResponseGenerator
        response_generator = ResponseGenerator(model_name="gpt2")

        # Example query
        query = "What is revenue?"
        logging.info(f"Processing query: {query}")

        # Retrieve relevant contexts and select top 3
        retrieved_contexts = retriever.retrieve(query, k=5)
        retrieved_contexts = retrieved_contexts[:3]  # Use only top 3 contexts

        # Truncate retrieved contexts to fit within the model's token limit
        logging.info("Truncating retrieved contexts to fit within model limits...")
        max_contexts = 3  # Adjust the number of contexts to include
        retrieved_contexts = retrieved_contexts[:max_contexts]

    

        # Generate response
        response = response_generator.generate_response(
            query=query, 
            retrieved_contexts=retrieved_contexts, 
            max_length=150, 
            temperature=0.7, 
            repetition_penalty=1.1
        )

        # Output response
        print("\nGenerated Response:", response)

        logging.info("RAG pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Critical error in RAG pipeline: {e}")
        raise CustomException("Failed to execute RAG pipeline.", e)

if __name__ == "__main__":
    main()
