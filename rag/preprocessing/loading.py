from hpack import struct
from yaml import FullLoader
# from pdf2image import convert_from_path, convert_from_bytes
# from pdf2image.exceptions import (
#     PDFInfoNotInstalledError,
#     PDFPageCountError,
#     PDFSyntaxError
# )

import fitz  # PyMuPDF
import cv2

import os 
import uuid
from dotenv import load_dotenv
from google import genai


from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# def pdf_to_image(path):
#     images = convert_from_path(path,dpi=200)
#     return images
# not using pdf2image anymore , using pymupdf for uniformity

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    page_images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for high-quality OCR and cropping
        pix = page.get_pixmap(dpi=300)
        img_path = f"temp_page_{page_num}.png"
        pix.save(img_path)
        page_images.append(img_path)
    return page_images
