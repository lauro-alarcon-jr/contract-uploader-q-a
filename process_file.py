import base64
import os

import streamlit as st
from google.cloud import documentai
from google.cloud import storage
from vertexai.language_models import ChatModel, InputOutputTextPair

# make a documentation of this code
#

# Set the environment variables for Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
documentai_client = documentai.DocumentProcessorServiceClient()
storage_client = storage.Client()
bucket_name = "exploration-contract-bucket"
processor_id = "8502f8a5b377cefc"
project_id = "coral-sum-399907"
location = "us"
bucket = storage_client.get_bucket(bucket_name)

# Cache for storing extracted text
extracted_text_cache = {}


# Function to process the uploaded PDF files
def process_files(files):
    extracted_text = []
    for file in files:
        # Check if extracted text is already cached
        if file.name in extracted_text_cache:
            text = extracted_text_cache[file.name]
        else:
            # Read the file content
            content = file.read()

            # Create a request object for the Document AI API
            request = {
                "name": f"projects/{project_id}/locations/{location}/processors/{processor_id}",
                "raw_document": {
                    "content": base64.b64encode(content).decode("utf-8"),
                    "mime_type": "application/pdf",
                }
            }

            # Call the Document AI API and get the response
            with st.spinner("Learning Contract. Please wait"):
                result = documentai_client.process_document(request)

            # Extract the document text and entities from the response
            document = result.document
            text = document.text

            # cache the extracted text
            extracted_text_cache[file.name] = text

        # Print the extracted text
        st.write(f"Extracted text from {file.name}:")
        st.write(text)
        st.write("---")

        extracted_text.append(text)

    return extracted_text

# chat model to be used for GenAI
chat_model = ChatModel.from_pretrained("chat-bison@001")


# Function to handle questions and answers
def questions_and_answer(question, extracted_text, temperature: float = 0.2) -> str:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.95,
        # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    chat = chat_model.start_chat(
        context="My name is Miles. You are a contract specialist, knowledgeable about contracts.",
        examples=[
            InputOutputTextPair(
                input_text="How much are the fees and other charges?",
                output_text="The fees and other charges are 100,000 pesos.",
            ),
        ],
    )

    # Combine all list of text into one string
    text = " ".join(extracted_text)

    # Combine text into one string the text output and the user input question
    combined_text = text + " " + question

    response = chat.send_message(
        combined_text, **parameters
    )

    # st.write(f"Response from Model: {response.text}")
    return response.text


# Streamlit web application
def main():
    # make st.title centered
    st.markdown("<h1 style='text-align: center;'>Lau.Ai</h1>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        extracted_text = process_files(uploaded_files)

        question = st.text_input("Enter your question")

        if question:
            with st.spinner("Answering your question. Please wait..."):
                # questions_and_answer(question, extracted_text)
                st.write(f"Response from Model: {questions_and_answer(question, extracted_text)}")


if __name__ == "__main__":
    main()
