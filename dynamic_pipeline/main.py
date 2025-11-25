import os
import json
import faiss
from sentence_transformers import SentenceTransformer

import sys
import os

# Add project root to PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.components.data_extraction import DataExtraction
from src.components.data_preprocessing import data_preprocessing_pipeline
from src.components.new_vector import VectorDBHandler
from src.components.retriever import Retriever
from src.components.response_generator import ResponseGenerator
from src.logger import logging
from src.exception import CustomException
import warnings




# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def main(file_path, query):

    try:
        logging.info("Starting RAG pipeline...")

        # Step 1: Data Extraction
        logging.info("Step 1: Extracting raw data from the uploaded PDF.")
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
        logging.info(f"Loaded cleaned text chunks: {len(cleaned_text_chunks)} chunks found.")

        # Generate text embeddings
        text_mapping, text_embeddings = vector_db_handler.generate_text_embeddings(cleaned_text_chunks)
        logging.info(f"Generated text embeddings: {len(text_embeddings)} embeddings created.")

        # Generate table embeddings
        flattened_table_folder = os.path.join("data", "processed", "flattened_table")
        table_mapping, table_embeddings = vector_db_handler.generate_table_embeddings(flattened_table_folder)
        logging.info(f"Generated table embeddings: {len(table_embeddings)} embeddings created.")

        # Combine mappings
        unified_mapping = text_mapping + table_mapping

        # Save unified mapping and create FAISS index
        vector_db_handler.save_unified_mapping(unified_mapping)
        vector_db_handler.create_faiss_index(unified_mapping, text_embeddings, table_embeddings)
        logging.info("Unified mapping and FAISS index created successfully.")

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
        retriever = Retriever(model=model, index=index, mapping_data=unified_mapping, similarity_threshold=1.2)  # Lowered similarity threshold

        # Initialize ResponseGenerator
        response_generator = ResponseGenerator(model_name="gpt2")

        # Example dynamic query for debugging
        #query = "What is revenue?"  # This will be replaced dynamically in app.py
        logging.info(f"Processing query: {query}")

        # Retrieve relevant contexts
        retrieved_contexts = retriever.retrieve(query, k=5)
        logging.info(f"Retrieved contexts: {len(retrieved_contexts)} contexts found.")
        for i, context in enumerate(retrieved_contexts):
            logging.info(f"Context {i+1}: {context}")

        # Truncate retrieved contexts to fit within model limits
        max_contexts = 3  # Adjust the number of contexts to include
        retrieved_contexts = retrieved_contexts[:max_contexts]
        logging.info(f"Truncated contexts to top {max_contexts}.")

        # Generate response
        try:
            response = response_generator.generate_response(
                query=query,
                retrieved_contexts=retrieved_contexts,
                max_length=150,  # Adjusted max_length for input length issue
                temperature=0.7,
                repetition_penalty=1.1
            )
            logging.info(f"Generated response: {response}")
            return response
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            raise CustomException("Failed to generate response.", e)

    except Exception as e:
        logging.error(f"Critical error in RAG pipeline: {e}")
        raise CustomException("Failed to execute RAG pipeline.", e)


