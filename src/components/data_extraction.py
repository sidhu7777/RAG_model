import os
import pdfplumber
import pandas as pd
from src.logger import logging
from src.exception import CustomException

class DataExtraction:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_output_dir = os.path.join("data", "raw", "text")
        self.table_output_dir = os.path.join("data", "raw", "table")
        self._create_directories()

    def _create_directories(self):
        """
        Create the text and table directories if they don't already exist.
        """
        try:
            os.makedirs(self.text_output_dir, exist_ok=True)
            os.makedirs(self.table_output_dir, exist_ok=True)
            logging.info(f"Created directories: {self.text_output_dir}, {self.table_output_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def extract_text(self):
        """
        Extract all text from the PDF and save it as a .txt file in the text directory.
        """
        try:
            with pdfplumber.open(self.file_path) as pdf:
                text = ''.join([page.extract_text() for page in pdf.pages])
            
            # Save text to a file
            text_file_path = os.path.join(self.text_output_dir, "extracted_text.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            
            logging.info(f"Extracted text saved to: {text_file_path}")
            return text_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def extract_tables(self):
        """
        Extract all tables from the PDF and save them as CSV files in the table directory.
        """
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if page.extract_tables():
                        for table_index, table in enumerate(page.extract_tables()):
                            # Convert the table to a DataFrame
                            df = pd.DataFrame(table)
                            # Save the table to a CSV file
                            table_file_path = os.path.join(self.table_output_dir, f"table_page_{i+1}_table_{table_index+1}.csv")
                            df.to_csv(table_file_path, index=False, header=False)
                            logging.info(f"Table saved to: {table_file_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        """
        Run the data extraction pipeline: extract text and tables.
        """
        try:
            logging.info("Starting data extraction process...")
            text_file_path = self.extract_text()
            self.extract_tables()
            logging.info("Data extraction completed successfully.")
            return text_file_path, self.table_output_dir
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example file path for testing
    file_path = r"C:\Users\91832\Desktop\RAG_Model_development\notebook\Sample Financial Statement.pdf"

    # Create an instance of DataExtraction and run the pipeline
    data_extractor = DataExtraction(file_path)
    text_path, table_dir = data_extractor.run()

    print(f"Extracted text saved to: {text_path}")
    print(f"Extracted tables saved in: {table_dir}")
