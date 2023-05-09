from paper_reader.abstract_reader import *
import os


class CERMINE_Reader(PaperReader):
    """assume we only have a directory that holds all our research papers, then we will only process once."""
    def __init__(self):
        self.executable_dir = '/Users/samuelyiin/recommender_system/paper_reader/CERMINE'
        self.paper_dir = '/Users/samuelyiin/recommender_system/papers_pdf/'
        # todo this will change later. not his responsibility
        self._parse_paper_to_zone_by_directory(self.paper_dir)

    def _parse_paper_to_zone_by_directory(self, paper_dir):
        change_dir_cmd = f'cd {self.executable_dir}'
        run_java_file_cmd = f'java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -outputs zones -path {paper_dir}'
        cmd = change_dir_cmd + ";" + run_java_file_cmd
        os.system(cmd)

    def parse(self, paper_fp) -> Dict:
        parsed_paper_content = self._read_processed_paper(paper_fp)
        return {'content': parsed_paper_content}

    def _read_processed_paper(self, paper_fp):
        # get processed paper name
        paper_name_parsed_dot = paper_fp.split('.')
        paper_name_parsed_dot[-1] = 'cermzones'
        processed_paper_fp = '.'.join(paper_name_parsed_dot)

        # read the processed paper into string
        with open(processed_paper_fp, 'r') as file:
            content = file.read()
            return content





