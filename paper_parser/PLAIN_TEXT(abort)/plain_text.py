from paper_parser.abstract_parser import *
import PyPDF2


class PlainTextReader(PaperReader):
    def parse_dir(self, raw_paper_dir, output_dict_dir) -> Dict:
        # Open the PDF file
        pdf_file = open(raw_paper_dir, 'rb')

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Loop through the pages and extract text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Close the PDF file
        pdf_file.close()

        return {'content': text}
