import chardet

file_path = "C:/Users/91832/Desktop/RAG_Model_development/vector_db/vector_db.index"

with open(file_path, "rb") as f:
    result = chardet.detect(f.read())
    print(f"Detected encoding: {result['encoding']}")
