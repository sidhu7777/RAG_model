import os
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import sys
import re
import numpy as np
from src.logger import logging
from src.exception import CustomException
import unicodedata

# Add wordninja path
sys.path.insert(0, r"C:\Users\91832\Desktop\RAG_Model_development\wordninja")
import wordninja

# Download required NLTK data
nltk.download('stopwords')

# Logger setup
logger = logging.getLogger("data_preprocessing")

# Define paths
RAW_TEXT_FOLDER = "data/raw/text"
RAW_TABLE_FOLDER = "data/raw/table"
PROCESSED_TEXT_FOLDER = "data/processed"
PROCESSED_TABLE_FOLDER = "data/processed/flattened_table"

# Ensure processed directories exist
os.makedirs(PROCESSED_TEXT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_TABLE_FOLDER, exist_ok=True)

# Text Data Preprocessing
def preprocess_text():
    try:
        logger.info("Starting text preprocessing...")
        combined_text_file = os.path.join(PROCESSED_TEXT_FOLDER, "text_final_output.txt")
        cleaned_text_chunks_file = os.path.join(PROCESSED_TEXT_FOLDER, "cleaned_text_chunks.json")

        # Combine all text files into one
        combined_text = ""
        for file_name in os.listdir(RAW_TEXT_FOLDER):
            file_path = os.path.join(RAW_TEXT_FOLDER, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                with open(file_path, "r", encoding="utf-8") as file:
                    combined_text += file.read() + "\n"

        # Save combined text
        with open(combined_text_file, "w", encoding="utf-8") as output_file:
            output_file.write(combined_text)
        logger.info(f"Combined text saved to {combined_text_file}")

        # Pre-clean the text to normalize spaces and remove artifacts
        cleaned_text = re.sub(r'\s+', ' ', combined_text)  # Replace multiple spaces/newlines with a single space
        cleaned_text = unicodedata.normalize("NFKD", cleaned_text).encode("ascii", "ignore").decode("utf-8")

        # Use WordNinja to split the cleaned text
        tokens = wordninja.split(cleaned_text)

        # Filter out unwanted tokens and single characters
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            token for token in tokens
            if token.lower() not in stop_words
            and token.strip()
            and len(token) > 1  # Ignore single-character tokens
        ]

        # Generate bi-grams and tri-grams
        bigrams = list(ngrams(filtered_tokens, 2))
        trigrams = list(ngrams(filtered_tokens, 3))

        # Chunk the cleaned text into pieces of size 1000
        chunk_size = 1000
        chunks = [filtered_tokens[i:i + chunk_size] for i in range(0, len(filtered_tokens), chunk_size)]

        # Save chunks to a JSON file
        with open(cleaned_text_chunks_file, "w", encoding="utf-8") as json_file:
            json.dump(chunks, json_file)
        logger.info(f"Cleaned text chunks saved to {cleaned_text_chunks_file}")

    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        raise CustomException(e)

# Table Data Preprocessing
def preprocess_table():
    try:
        logger.info("Starting table preprocessing...")
        for file_name in os.listdir(RAW_TABLE_FOLDER):
            file_path = os.path.join(RAW_TABLE_FOLDER, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                try:
                    # Read table
                    table = pd.read_csv(file_path)

                    # Handle missing values
                    table = table.replace(["", " ", "NULL", "null", np.nan], np.nan)

                    # Flatten the table (assume no nested structures for now)
                    # Save cleaned and flattened table
                    flattened_file_path = os.path.join(PROCESSED_TABLE_FOLDER, file_name)
                    table.to_csv(flattened_file_path, index=False)

                    logger.info(f"Processed and saved table: {flattened_file_path}")
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {e}")
                    raise CustomException(e)
    except Exception as e:
        logger.error(f"Error in table preprocessing: {e}")
        raise CustomException(e)

# Pipeline
def data_preprocessing_pipeline():
    try:
        logger.info("Data preprocessing pipeline started.")
        preprocess_text()
        preprocess_table()
        logger.info("Data preprocessing pipeline completed.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise CustomException(e)

if __name__ == "__main__":
    data_preprocessing_pipeline()
