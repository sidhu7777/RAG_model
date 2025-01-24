import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.logger import logging
from src.exception import CustomException
from src.components.handling_embedding import EmbeddingHandler  # Import the EmbeddingHandler class

class VectorDBHandler:
    def __init__(self, model_name="all-MiniLM-L6-v2", vector_db_path=None, embedding_folder=None):
        try:
            # Initialize EmbeddingHandler
            self.embedding_handler = EmbeddingHandler(model_name=model_name)
            
            # Paths for vector database and embedding folder
            self.vector_db_path = vector_db_path or os.path.join(os.getcwd(), "vector_db")
            self.embedding_folder = embedding_folder or os.path.join(os.getcwd(), "embeddings")
            os.makedirs(self.vector_db_path, exist_ok=True)
            os.makedirs(self.embedding_folder, exist_ok=True)
        except Exception as e:
            logging.error(f"Error initializing VectorDBHandler: {e}")
            raise CustomException(e)

    def generate_text_embeddings(self, cleaned_text_chunks):
        try:
            logging.info("Generating text embeddings via EmbeddingHandler...")
            text_embeddings, text_mapping = self.embedding_handler.generate_text_embeddings(cleaned_text_chunks)
            
            # Save text embeddings for validation/debugging
            text_embeddings_path = os.path.join(self.embedding_folder, "cleaned_text_chunks_embeddings.npy")
            np.save(text_embeddings_path, text_embeddings)
            logging.info(f"Text embeddings saved to {text_embeddings_path}")
            return text_mapping, text_embeddings
        except Exception as e:
            logging.error(f"Error generating text embeddings: {e}")
            raise CustomException(e)

    def generate_table_embeddings(self, flattened_table_folder):
        try:
            logging.info("Generating table embeddings via EmbeddingHandler...")
            
            # Load all CSV files in the folder
            table_files = [
                (os.path.basename(file), pd.read_csv(os.path.join(flattened_table_folder, file)))
                for file in os.listdir(flattened_table_folder)
                if file.endswith(".csv")
            ]
            
            # Delegate embedding generation to EmbeddingHandler
            table_embeddings, table_mapping = self.embedding_handler.generate_table_embeddings(table_files)
            
            # Save table embeddings for validation/debugging
            for metadata, embedding in zip(table_mapping, table_embeddings):
                embedding_file = os.path.join(
                    self.embedding_folder,
                    f"{metadata['file_name'].replace('.csv', '')}_row_{metadata['row_index']}_embedding.npy"
                )
                np.save(embedding_file, embedding)
                logging.info(f"Saved embedding for row {metadata['row_index']} in {metadata['file_name']} to {embedding_file}")
            
            return table_mapping, np.vstack(table_embeddings)
        except Exception as e:
            logging.error(f"Error generating table embeddings: {e}")
            raise CustomException(e)

    def create_faiss_index(self, unified_mapping, text_embeddings, table_embeddings):
        try:
            logging.info("Creating FAISS index from precomputed embeddings...")

            # Combine text and table embeddings
            all_embeddings = np.vstack([text_embeddings, table_embeddings])

            # Validate embedding count matches mapping entries
            if len(all_embeddings) != len(unified_mapping):
                raise ValueError("Mismatch between FAISS embeddings and unified mapping entries.")

            # Create FAISS index
            embedding_dim = all_embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(all_embeddings)

            # Save FAISS index
            index_path = os.path.join(self.vector_db_path, "vector_index.faiss")
            faiss.write_index(index, index_path)
            logging.info(f"FAISS index created with {index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
            raise CustomException(e)

    def validate_index_and_mapping(self, unified_mapping):
        try:
            index_path = os.path.join(self.vector_db_path, "vector_index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError("FAISS index file not found.")
            index = faiss.read_index(index_path)

            total_vectors = index.ntotal
            total_metadata = len(unified_mapping)

            logging.info(f"FAISS index vectors: {total_vectors}")
            logging.info(f"Unified mapping entries: {total_metadata}")

            if total_vectors != total_metadata:
                logging.warning(f"Mismatch detected! FAISS vectors: {total_vectors}, Metadata entries: {total_metadata}")
                for entry in unified_mapping[:5]:  # Print first few entries for debugging
                    logging.debug(f"Unified mapping entry: {entry}")
                return False

            logging.info("FAISS vectors and metadata entries are synchronized.")
            return True
        except Exception as e:
            logging.error(f"Error validating FAISS index and unified mapping: {e}")
            raise CustomException(e)

if __name__ == "__main__":
    try:
        logging.info("Starting embedding pipeline...")
        vector_db_handler = VectorDBHandler()

        # Load cleaned text chunks
        text_chunks_path = os.path.join(os.getcwd(), "data", "processed", "cleaned_text_chunks.json")
        with open(text_chunks_path, "r", encoding="utf-8") as f:
            cleaned_text_chunks = json.load(f)

        # Generate text embeddings
        text_mapping, text_embeddings = vector_db_handler.generate_text_embeddings(cleaned_text_chunks)

        # Generate table embeddings
        flattened_table_folder = os.path.join(os.getcwd(), "data", "processed", "flattened_table")
        table_mapping, table_embeddings = vector_db_handler.generate_table_embeddings(flattened_table_folder)

        # Combine mappings
        unified_mapping = text_mapping + table_mapping

        # Save unified mapping
        with open("enhanced_unified_mapping.json", "w") as file:
            json.dump(unified_mapping, file, indent=4)

        # Create FAISS index
        vector_db_handler.create_faiss_index(unified_mapping, text_embeddings, table_embeddings)

        # Validate FAISS index and unified mapping
        is_valid = vector_db_handler.validate_index_and_mapping(unified_mapping)
        if is_valid:
            logging.info("Pipeline executed successfully and validated.")
        else:
            logging.warning("Pipeline completed but validation failed.")
    except Exception as e:
        logging.error(f"Error in the embedding pipeline: {e}")
