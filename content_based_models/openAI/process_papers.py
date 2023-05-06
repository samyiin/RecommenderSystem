import PyPDF2
import re
import sys
import os
print(sys.path)
print(os.getcwd())
print(os.path)


def process_paper(paper_path):
    # Open the PDF file and extract the text
    pdf_file = open(paper_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ''
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # Split the text into sections based on keywords
    content = pdf_text.split('Introduction')[1]

    # Clean up the extracted text
    content = content.replace('\n', ' ').strip()

    # Create a dictionary with the extracted information
    paper_dict = {
        'Content': content
    }
    return paper_dict

x = process_paper('../../research_paper_pdf/421425.pdf')
print(1)