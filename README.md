# RAG Model for Financial QA Bot

## Overview

This project is a **Retrieval-Augmented Generation (RAG) Model** designed for answering financial queries related to **Profit & Loss (P&L) statements** extracted from PDF documents. The project includes a backend pipeline for data processing and a user-friendly interactive interface for querying financial data. It is developed to demonstrate cutting-edge AI capabilities in financial data extraction and question answering, making it a valuable addition to a professional portfolio.

## Features

1. **Data Extraction and Preprocessing**

   - Extracts P&L tables from PDFs using Camelot and pdfplumber.
   - Cleans and converts extracted data into structured formats (e.g., Pandas DataFrames).

2. **RAG Model Pipeline**

   - Embeds financial terms and metrics using Sentence Transformers.
   - Stores embeddings in FAISS, a high-performance vector database for efficient retrieval.
   - Implements a RAG-based pipeline to retrieve relevant financial information and generate human-like answers.

3. **Interactive Interface**

   - Built with Streamlit, allowing users to:
     - Upload PDF documents containing P&L statements.
     - Query financial data in natural language (e.g., "What is the gross profit for Q3 2024?").
     - View relevant financial data alongside generated answers.

4. **Performance Optimization**
   - Handles large P&L documents and multiple queries efficiently.
   - Ensures accurate parsing and retrieval of financial terms.

---

## Project Structure

```
RAG_Model_Development/
├── app/
│   └── app.py                    # Streamlit-based user interface
├── data/
│   ├── raw/                      # Raw PDF files
│   └── processed/                # Preprocessed financial data
├── logs/                         # Logs for debugging and performance monitoring
├── notebooks/
│   └── Financial_bot_changing__again.ipynb  # Development and experimentation
├── src/
│   ├── data_extraction.py        # PDF parsing and data cleaning
│   ├── embedding_generator.py    # Embedding generation and FAISS integration
│   ├── retriever.py              # RAG pipeline for retrieval
│   └── response_generator.py     # Language model for generating answers
├── tests/
│   └── test_pipeline.py          # Unit tests for the RAG model pipeline
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
└── setup.py                      # Package setup file
```

---

## How It Works

1. **Data Extraction**:

   - Upload a PDF document containing P&L statements.
   - Extract tabular data using Camelot or pdfplumber.

2. **Embedding and Retrieval**:

   - Generate embeddings for extracted financial terms using Sentence Transformers.
   - Store embeddings in a FAISS vector database.
   - Retrieve relevant data segments based on user queries.

3. **Answer Generation**:

   - Use a pre-trained Hugging Face transformer model to generate responses from retrieved data.
   - Display the relevant data alongside the generated answer in the interface.

4. **User Interaction**:
   - Query examples:
     - "What are the total expenses for Q2 2023?"
     - "Show the operating margin for the past 6 months."
   - Receive accurate and contextually relevant answers in real time.

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/RAG_Model_Development.git
   cd RAG_Model_Development
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install Ghostscript (required for Camelot):
   - [Download Ghostscript](https://ghostscript.com/download/gsdnld.html)
   - Add Ghostscript to your system's PATH.

---

## Example Queries

Here are some examples of financial queries the QA bot can handle:

- **"What is the gross profit for Q3 2024?"**
- **"How do the net income and operating expenses compare for Q1 2024?"**
- **"What are the total expenses for the past year?"**

---

## Challenges and Solutions

1. **PDF Parsing Variability**:

   - Different P&L formats required robust handling with tools like Camelot and pdfplumber.

2. **Embedding Accuracy**:

   - Fine-tuned Sentence Transformers to enhance retrieval performance.

3. **Latency Optimization**:
   - Used FAISS for efficient nearest-neighbor searches, reducing response time.

---

## Future Enhancements

- Deploy on cloud platforms like AWS or Azure for broader accessibility.
- Implement role-based access control for enhanced security.
