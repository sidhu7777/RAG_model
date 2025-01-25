import json
from transformers import pipeline
from src.logger import logging
from src.exception import CustomException


class ResponseGenerator:
    def __init__(self, model_name="gpt2", device="cpu"):
        """
        Initialize the Response Generator with a Hugging Face model.
        Args:
            model_name (str): Name of the Hugging Face model.
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        try:
            logging.info(f"Loading response generation model: {model_name}")
            self.generator = pipeline("text-generation", model=model_name, device=0 if device == "cuda" else -1)
            logging.info("ResponseGenerator initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing ResponseGenerator: {e}")
            raise CustomException("Failed to initialize ResponseGenerator.", e)

    def generate_response(self, query, retrieved_contexts, max_length=150, temperature=0.7, repetition_penalty=1.2):
        """
        Generate a response based on the query and retrieved contexts.
        Args:
            query (str): User's query.
            retrieved_contexts (list of tuples): Retrieved contexts [(content, distance), ...].
            max_length (int): Maximum length of the generated response.
            temperature (float): Sampling temperature.
            repetition_penalty (float): Penalty for repeated tokens.
        Returns:
            str: Generated response.
        """
        try:
            logging.info(f"Generating response for query: {query}")

            # Extract meaningful content from retrieved contexts
            context = "\n".join([content for content, _ in retrieved_contexts if content != "No content available"])
            
            # Add fallback logic for empty or irrelevant context
            if not context.strip():
                logging.warning("No meaningful context retrieved. Using fallback response.")
                return "I couldn't retrieve relevant information. Please refine your query."

            logging.info(f"Generated context: {context}")

            # Create the prompt
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            logging.info(f"Generated prompt: {prompt}")

            # Generate the response
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
            )
            answer = response[0]["generated_text"]
            logging.info("Response generated successfully.")
            return answer.strip()
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            raise CustomException("Failed to generate response.", e)


if __name__ == "__main__":
    try:
        # Load retrieved results from file
        retriever_output_path = "retriever_output.json"
        with open(retriever_output_path, "r", encoding="utf-8") as file:
            retrieved_results = json.load(file)
        logging.info(f"Retrieved results loaded from {retriever_output_path}")

        # Log retrieved contexts for debugging
        logging.info(f"Retrieved contexts: {retrieved_results}")

        # Initialize ResponseGenerator
        generator = ResponseGenerator(model_name="gpt2")

        # Example query
        query = "What is revenue?"

        # Generate response
        response = generator.generate_response(query=query, retrieved_contexts=retrieved_results)
        print("Generated Response:", response)
    except Exception as e:
        logging.error(f"Critical error in ResponseGenerator: {e}")
