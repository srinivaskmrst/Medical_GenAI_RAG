"""Document loading utilities."""
import os
from typing import List
import docx
from pypdf import PdfReader
from dotenv import load_dotenv
from docx import Document
import json
from PIL import Image
import pytesseract


class DocumentLoader:
    """Loader for PDF files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def IdentifyFileType(self) -> str:
        """Identify the file type based on the file extension."""
        _, file_extension = os.path.splitext(self.file_path)
        return file_extension.lower()


    def CallingLoader(self) -> List[str]:
        """Call the appropriate loader based on the file type."""
        file_type = self.IdentifyFileType()
        if file_type == ".pdf":
            return self.Pdfload()
        elif file_type == ".docx":
            return self.Docxload()
        elif file_type == ".json":
            return self.Jsonload()
        elif file_type == ".md":
            return self.Mdload()
        elif file_type in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return self.ImageLoad()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")


    def Pdfload(self) -> List[str]:
        """Load the PDF file and return its text content as a list of strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        reader = PdfReader(self.file_path)
        text_content = []
        for page in reader.pages:
            text_content.append(page.extract_text())
        return text_content

    def Docxload(self) -> List[str]:
        """Load the DOCX file and return its text content as a list of strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        doc = Document(self.file_path)
        text_content = []
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        return text_content

    def Jsonload(self) -> List[str]:
        """Load the JSON file and return its text content as a list of strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

       
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [str(data)]

    def Mdload(self) -> List[str]:
        """Load the Markdown file and return its text content as a list of strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            text_content = f.readlines()
        return text_content

    def ImageLoad(self) -> List[str]:
        """Load the image file and return its text content as a list of strings."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        # Open image using PIL
        with Image.open(self.file_path) as img:
            # Extract text from the PIL image using pytesseract
            raw_text = pytesseract.image_to_string(img)

        # Convert extracted text into a list of non-empty lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return lines
