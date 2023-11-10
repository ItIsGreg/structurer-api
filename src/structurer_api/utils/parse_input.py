from PyPDF2 import PdfReader
from PIL import Image
import pytesseract


def extract_pdf_text(file):
    pdfReader = PdfReader(file)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text


def extract_scan_text(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text
