import streamlit as st
import os
from dynamic_pipeline.main import main

# Streamlit App Layout
st.title("RAG-Based Financial QA Bot")
st.write("Upload a financial statement and ask your query.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")
if uploaded_file:
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded successfully: {uploaded_file.name}")

    # Query Input Section
    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("Processing your query..."):
            try:
                # Pass the uploaded file and query to main
                response = main(file_path, query)

                st.success("Processing completed successfully!")

                # Display Results
                if response:
                    st.write("### Generated Response")
                    st.success(response)
                else:
                    st.warning("No response generated. Please refine your query.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Footer
st.markdown("---")
st.markdown("Created by [Vineeth Raja Banala]")

