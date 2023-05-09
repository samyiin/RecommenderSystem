from paper_reader.abstract_reader import *
import PyPDF2


class PlainTextReader(PaperReader):
    def parse(self, paper_fp) -> Dict:
        # Open the PDF file
        pdf_file = open(paper_fp, 'rb')

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Loop through the pages and extract text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Close the PDF file
        pdf_file.close()

        return {'content': text}
