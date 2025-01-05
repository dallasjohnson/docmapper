import os
from dotenv import load_dotenv
import RAGApplication
from google import genai
import streamlit as st

def main():
    # Load environment variables
    load_dotenv()

    # Page title
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    # Test the API key
    try:
        test_client = genai.Client(api_key=api_key)
        test_response = test_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Hello, this is a test message."
        )
        print("API key is working!",test_response.text)
    except Exception as e:
        print(f"API test failed: {e}")
        raise ValueError("Invalid API key.")

    # Form
    with st.form(key="stimy_form"):
        pdf_path = st.file_uploader("Upload a PDF file", type=["pdf"])  
        questions = st.text_input('Enter your question:', placeholder='Please provide a short summary.')
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and pdf_path and questions:
        try:
            # Save the uploaded PDF to a temp file
            temp_pdf_path = f"temp_{pdf_path.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_path.getbuffer())
            
            # Initialize application
            app = RAGApplication(api_key)
            
            # Process PDF and answer questions
            print(f"Processing PDF: {pdf_path.name}")
            with st.spinner("Thinking..."):
                app.process_pdf(temp_pdf_path)
                answers = app.answer_questions(questions)

            # Display answers
            for result in answers:
                print(f"Question: {result['question']}")
                print(f"Answer: {result['answer']}")
                print(f"Source: {result['source']}")
                print("-" * 80)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "main":
    main()
