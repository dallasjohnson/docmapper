import textwrap
import time
import types
import numpy as np
import pandas as pd
from ratelimit import sleep_and_retry, limits
from config import Config
from PIL import Image
from google import genai
import streamlit as st

class GeminiClient:
    """Client for Gemini API"""

    def __init__(self, api_key: str):
        """Initialize the Gemini client"""

        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
    
    def make_prompt(self, element: str) -> str:
        """Create prompt for summarization"""

        return f"""You are an agent tasked with summarizing research tables and texts from research papers for retrieval. 
                  These summaries will be embedded and used to retrieve the raw text or table elements. 
                  Give a concise summary of the tables or text that is well optimized for retrieval. 
                  Table or text: {element}"""
    
    def analyze_page(self, image: Image.Image) -> str:
        """Analyze a page of a research paper"""

        prompt = """You are an assistant tasked with summarizing images for retrieval. 
                   These summaries will be embedded and used to retrieve the raw image.
                   Give a concise summary of the image that is well optimized for retrieval.
                   If it's a table, extract all elements of the table.
                   If it's a graph, explain the findings in the graph.
                   Include details about color, proportion, and shape if necessary to describe the image.
                   Extract all text content from the page accurately.
                   Do not include any numbers that are not mentioned in the image."""
        
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=[prompt, image]
            )
            return response.text if response.text else ""
        except Exception as e:
            st.write(f"Error analyzing page: {e}")
            return ""
    
    @sleep_and_retry
    @limits(calls=30, period=60)
    def create_embeddings(self, data: str) -> str:
        """Create embeddings with rate limiting - exactly as in Google's example"""

        time.sleep(1)

        return self.client.models.embed_content(
        model=Config.TEXT_EMBEDDING_MODEL_ID,
        contents=data,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
    
    def find_best_passage(self, query: str, dataframe: pd.DataFrame) -> str:
        """Find the best passage for a query"""

        try:
            query_embedding = self.client.models.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )

            dot_product_scores = np.dot(
                np.stack(dataframe['Embeddings']),
                query_embedding.embeddings[0].values
            )

            idx = np.argmax(dot_product_scores)
            content = dataframe.iloc[idx]['Original Content']

            return {
                'page': content['page_number'],
                'content': content['content']
            }

        except Exception as e:
            st.write(f"Error finding best passage: {e}")
            return ""

    def make_answer_prompt(self, query: str, passage: dict) -> str:
            """Create prompt for answering questions"""
            escaped = passage['content'].replace("'", "").replace('"', "").replace("\n", " ")
            return textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
                                    You are answering questions about a research paper. 
                                    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
                                    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                                    strike a friendly and conversational tone. 
                                    If the passage is irrelevant to the answer, you may ignore it.
                                    
                                    QUESTION: '{query}'
                                    PASSAGE: '{passage}'
                                    
                                    ANSWER:
                                """).format(query=query, passage=escaped)