from PIL import Image
from typing import List
from config import Config
import pymupdf
import io

class PDFProcessor:
    """Handles PDF processing using PyMuPDF and Gemini's vision capabilities"""

    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = Config.DPI) -> List[Image.Image]:
        """Converts a PDF file to a list of PIL images"""

        images = []
        with pymupdf.open(pdf_path) as pdf:
            for page_number in range(pdf.page_count):
                page = pdf[page_number]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72))
                # images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.tobytes()))
                
                # Convert PyMuPDF pixmap to PIL Image
                image_data = pix.tobytes("png")
                images.append(Image.open(io.BytesIO(image_data)))

        return images