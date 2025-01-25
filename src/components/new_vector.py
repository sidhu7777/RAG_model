import os
import json
import faiss
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.handling_embedding import EmbeddingHandler


class VectorDBHandler:
    def __init__(self, model_name="all-MiniLM-L6-v2", vector_db_path=None, embedding_folder=None):
        try:
            self.embedding_handler = EmbeddingHandler(model_name=model_name)
            self.vector_db_path = vector_db_path or os.path.join(os.getcwd(), "vector_db")
            self.embedding_folder = embedding_folder or os.path.join(os.getcwd(), "embeddings")
            os.makedirs(self.vector_db_path, exist_ok=True)
            os.makedirs(self.embedding_folder, exist_ok=True)
        except Exception as e:
            logging.error(f"Error initializing VectorDBHandler: {e}")
            raise CustomException(e)

    def generate_text_embeddings(self, cleaned_text_chunks):
        try:
            logging.info("Generating text embeddings...")
            text_mapping, text_embeddings = self.embedding_handler.generate_text_embeddings(cleaned_text_chunks)

            # Save text embeddings
            text_embeddings_path = os.path.join(self.embedding_folder, "cleaned_text_chunks_embeddings.npy")
            np.save(text_embeddings_path, text_embeddings)
            logging.info(f"Text embeddings saved to {text_embeddings_path}")
            return text_mapping, text_embeddings
        except Exception as e:
            logging.error(f"Error generating text embeddings: {e}")
            raise CustomException(e)

    def generate_table_embeddings(self, flattened_table_folder):
        try:
            logging.info("Generating table embeddings...")
            table_files = [
                (os.path.basename(file), pd.read_csv(os.path.join(flattened_table_folder, file)))
                for file in os.listdir(flattened_table_folder)
                if file.endswith(".csv")
            ]
            table_mapping, table_embeddings = self.embedding_handler.generate_table_embeddings(table_files)

            for metadata, embedding in zip(table_mapping, table_embeddings):
                embedding_file = os.path.join(
                    self.embedding_folder,
                    f"{metadata['file_name'].replace('.csv', '')}_row_{metadata['row_index']}_embedding.npy"
                )
                np.save(embedding_file, embedding)
                logging.info(f"Saved embedding for row {metadata['row_index']} in {metadata['file_name']} to {embedding_file}")
            return table_mapping, table_embeddings
        except Exception as e:
            raise CustomException("Error generating table embeddings", e)

    def create_faiss_index(self, unified_mapping, text_embeddings, table_embeddings):
        try:
            logging.info("Creating FAISS index...")
            all_embeddings = np.vstack([text_embeddings, table_embeddings])
            if len(all_embeddings) != len(unified_mapping):
                raise ValueError("Mismatch between embeddings and unified mapping entries.")
            embedding_dim = all_embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(all_embeddings)
            index_path = os.path.join(self.vector_db_path, "vector_index.faiss")
            faiss.write_index(index, index_path)
            logging.info(f"FAISS index created with {index.ntotal} vectors.")
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
            raise CustomException(e)

    def save_unified_mapping(self, unified_mapping):
        try:
            mapping_path = os.path.join(self.vector_db_path, "vector_db.index")
            with open(mapping_path, "w", encoding="utf-8") as file:
                json.dump(unified_mapping, file, indent=4)
            logging.info(f"Unified mapping saved to {mapping_path}")
        except Exception as e:
            logging.error(f"Error saving unified mapping: {e}")
            raise CustomException(e)
