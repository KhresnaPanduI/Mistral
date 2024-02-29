from PyPDF2 import PdfReader


def parse_pdf_to_text(pdf_file):
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        