import os
import time
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from GeminiClient import GeminiClient
from PDFProcessor import PDFProcessor
from config import Config
import streamlit as st
class RAGApplication:
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key)
        self.data_df = None

    def process_pdf(self, pdf_path: str):
        """Process PDF using Gemini's vision capabilities"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        # Convert PDF pages to images
        images = PDFProcessor.pdf_to_images(pdf_path)

        # Analyze each page
        page_contents = []
        page_analyses = []

        st.write("Analyzing PDF pages...")
        for i, image in enumerate(images):
            content = self.gemini_client.analyze_page(image)
            if content:
                page_contents.append({
                    'page_number': i+1,
                    'content': content
                })
                page_analyses.append(content)
            
        if not page_analyses:
            raise ValueError("No content found in PDF")
        
        self.data_df = pd.DataFrame({
            'Original Content': page_contents, 
            'Analysis': page_analyses
            })
        
        # Generate embeddings
        st.write("\nGenerating embeddings...")
        embeddings = []

        try:
            for text in tqdm(self.data_df['Analysis']):
                embeddings.append(self.gemini_client.create_embeddings(text))
        except Exception as e:
            st.write(f"Error generating embeddings: {e}")
            time.sleep(10)
        
        _embeddings = []
        for embedding in embeddings:
            _embeddings.append(embedding.embeddings[0].values)

        self.data_df['Embeddings'] = embeddings

    def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """Answer a list of questions using the processed data"""
        if self.data_df is None:
            raise ValueError("Please process a PDF first using process_pdf()")
            
        answers = []
        for question in questions:
            try:
                passage = self.gemini_client.find_best_passage(question, self.data_df)
                prompt = self.gemini_client.make_answer_prompt(question, passage)
                response = self.gemini_client.client.models.generate_content(
                    model=Config.MODEL_NAME,
                    contents=prompt
                )
                answers.append({
                    'question': question,
                    'answer': response.text,
                    'source': f"Page {passage['page']}\nContent: {passage['content']}"
                })
            except Exception as e:
                st.write(f"Error processing question '{question}': {e}")
                answers.append({
                    'question': question,
                    'answer': f"Error generating answer: {str(e)}",
                    'source': "Error"
                })
            
        return answers
