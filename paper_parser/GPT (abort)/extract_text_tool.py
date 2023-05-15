# need to pip install PyMuPDF
import fitz
import PyPDF2


def get_pure_text(pdf_fp):
    # Open the PDF file
    pdf_file = open(pdf_fp, 'rb')

    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Loop through the pages and extract text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Close the PDF file
    pdf_file.close()

    return text

