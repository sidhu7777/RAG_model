import json

# Path to cleaned text chunks
text_chunk_path = r"C:\Users\91832\Desktop\RAG_Model_development\data\processed\cleaned_text_chunks.json"

with open(text_chunk_path, "r", encoding="utf-8") as file:
    text_chunks = json.load(file)

print(f"Total text chunks: {len(text_chunks)}")
for i, chunk in enumerate(text_chunks[:5]):  # Print first 5 chunks
    print(f"Chunk {i}: {chunk[:50]}...")  # Display first 50 characters
