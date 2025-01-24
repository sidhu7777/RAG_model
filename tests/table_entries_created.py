import os
import pandas as pd

# Path to flattened tables folder
table_folder_path = r"C:\Users\91832\Desktop\RAG_Model_development\data\processed\flattened_table"

# Count rows in all flattened tables
total_rows = 0
for file_name in os.listdir(table_folder_path):
    file_path = os.path.join(table_folder_path, file_name)
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
        total_rows += len(df)

print(f"Number of table entries created: {total_rows}")

