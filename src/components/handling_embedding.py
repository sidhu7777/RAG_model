import numpy as np
from sentence_transformers import SentenceTransformer
from src.logger import logging
from src.exception import CustomException
import pandas as pd

class EmbeddingHandler:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embedding(self, text):
        """
        Generate embeddings for a single piece of text.
        Args:
            text (str): Input text for which to generate embeddings.
        Returns:
            np.ndarray: Generated embedding as a numpy array.
        """
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logging.error(f"Error generating embedding for text: {e}")
            raise CustomException(f"Embedding generation failed for text: {text}", e)

    def generate_text_embeddings(self, cleaned_text_chunks):
        """
        Generate embeddings for cleaned text chunks.
        Args:
            cleaned_text_chunks (list of list): List of text chunks, where each chunk is a list of tokens.
        Returns:
            tuple: (text_mapping, text_embeddings)
                - text_mapping: List of dictionaries with metadata for each text chunk.
                - text_embeddings: Numpy array of embeddings for all text chunks.
        """
        try:
            logging.info("Generating embeddings for text chunks...")
            text_embeddings = []
            text_mapping = []

            # Iterate over each text chunk
            for idx, chunk in enumerate(cleaned_text_chunks):
                # Join the chunk into a single string
                chunk_text = " ".join(chunk)

                # Generate embedding for the chunk
                embedding = self.generate_embedding(chunk_text)

                # Append embedding and metadata
                text_embeddings.append(embedding)
                text_mapping.append({"chunk_index": idx, "content": chunk_text})

            # Debugging: Check types
            logging.info("Type of text_mapping inside generate_text_embeddings: %s", type(text_mapping))
            logging.info("Type of text_embeddings inside generate_text_embeddings: %s", type(text_embeddings))

            # Return text_mapping as a list of dictionaries and embeddings as a numpy array
            return text_mapping, np.vstack(text_embeddings)

        except Exception as e:
            logging.error(f"Error generating text embeddings: {e}")
            raise CustomException("Failed to generate text embeddings.", e)

    def generate_table_embeddings(self, table_files):
        """
        Generate embeddings for tables by processing each row of the DataFrame.
        Args:
            table_files (list): List of tuples (file_name, DataFrame) containing table data.
        Returns:
            tuple: (table_mapping, table_embeddings) where:
                - table_mapping is a list of dictionaries with metadata for each row.
                - table_embeddings is a numpy array of embeddings for all rows.
        """
        try:
            logging.info("Generating embeddings for table rows...")
            table_embeddings = []
            table_mapping = []

            # Process each file and its rows
            for file_name, df in table_files:
                for row_index, row in df.iterrows():
                    # Convert row to a single string for embedding
                    row_content = " | ".join(map(str, row.tolist()))

                    # Generate embedding for the row
                    embedding = self.generate_embedding(row_content)

                    # Append embedding and metadata
                    table_embeddings.append(embedding)
                    table_mapping.append({
                        "file_name": file_name,
                        "row_index": row_index,
                        "row_content": row_content,
                    })

            # Debugging: Check types
            logging.info("Type of table_mapping inside generate_table_embeddings: %s", type(table_mapping))
            logging.info("Type of table_embeddings inside generate_table_embeddings: %s", type(table_embeddings))

            # Return table_mapping as a list and embeddings as a numpy array
            return table_mapping, np.vstack(table_embeddings)

        except Exception as e:
            logging.error(f"Error generating table embeddings: {e}")
            raise CustomException("Failed to generate table embeddings.", e)
