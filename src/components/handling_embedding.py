import numpy as np
from sentence_transformers import SentenceTransformer
from src.logger import logging
from src.exception import CustomException
import pandas as pd

class EmbeddingHandler:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        try:
            logging.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error(f"Error initializing EmbeddingHandler: {e}")
            raise CustomException(e)

    def generate_text_embeddings(self, chunks):
        try:
            logging.info(f"Generating embeddings for {len(chunks)} text chunks.")
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            metadata = [{"chunk_index": i} for i in range(len(chunks))]
            return embeddings, metadata
        except Exception as e:
            logging.error(f"Error generating text embeddings: {e}")
            raise CustomException(e)

    def generate_table_embeddings(self, tables):
        try:
            logging.info(f"Generating embeddings for {len(tables)} flattened tables.")
            embeddings = []
            metadata = []
            for file_name, table in tables:
                for _, row in table.iterrows():
                    row_text = " ".join([str(value) for value in row])
                    embedding = self.model.encode([row_text])[0]
                    embeddings.append(embedding)
                    metadata.append({"file_name": file_name, "row_index": row.name})
            return np.array(embeddings), metadata
        except Exception as e:
            logging.error(f"Error generating table embeddings: {e}")
            raise CustomException(e)
