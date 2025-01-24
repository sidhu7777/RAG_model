import json

# Path to cleaned text chunks
text_chunk_path = r"C:\Users\91832\Desktop\RAG_Model_development\data\processed\cleaned_text_chunks.json"

# Load and count text chunks
with open(text_chunk_path, "r", encoding="utf-8") as file:
    text_chunks = json.load(file)

print(f"Number of text chunks created: {len(text_chunks)}")