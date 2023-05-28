from paper_parser.abstract_parser import *
import os


class CERMINEParser(PaperReader):
    def __init__(self, cermine_path, ):
        # find the directory for java executable
        project_root_directory = "recommender_system"
        relative_path = 'paper_parser/CERMINE'
        cwd = os.getcwd().split('/')
        content_root_temp = []
        for directory in cwd:
            content_root_temp.append(directory)
            if directory == project_root_directory:
                break
        # assume it will not be empty
        content_root = '/'.join(content_root_temp)
        self.executable_dir = os.path.join(content_root, relative_path)

    def _parse_paper_to_zone_by_directory(self, paper_dir):
        change_dir_cmd = f'cd {self.executable_dir}'
        run_java_file_cmd = f'java -cp cermine-impl-1.13-jar-with-dependencies.jar pl.edu.icm.cermine.ContentExtractor -outputs zones -path {paper_dir}'
        cmd = change_dir_cmd + ";" + run_java_file_cmd
        os.system(cmd)

    def _parse_zone_to_attribute_dict(self, zone_file_path):
        # read the processed paper into string
        with open(zone_file_path, 'r') as file:
            content = file.read()
        return {'content': content}

    def parse_dir(self, raw_paper_dir, output_dict_dir) -> None:
        self._parse_paper_to_zone_by_directory(raw_paper_dir)
        # find all zone files and write them into attribute dict
        zone_files = []
        for root, _, filenames in os.walk(raw_paper_dir):
            for filename in filenames:
                '''can handle subdirectory'''
                if filename.endswith(".cermzones"):
                    # parse zone files to attr dict
                    zone_file_path = os.path.join(root, filename)
                    zone_files.append(zone_file_path)
                    attribute_dict = self._parse_zone_to_attribute_dict(zone_file_path)
                    # write attr dict to file
                    output_filename_temp = filename.split('.')[:-1]
                    output_filename_temp.append('json')
                    output_filename = '.'.join(output_filename_temp)
                    output_filepath = os.path.join(output_dict_dir, output_filename)
                    self.write_attribute_dict_to_file(attribute_dict, output_filepath)


# how to use
reader = CERMINEParser()
raw_paper_dir = 'paper_parser/simulation_papers/papers_pdf'
zone_file_path = '/database/papers_pdf/test2.cermzones'
output_dict_dir = "/database/papers_attr_dict"
reader.parse_dir(raw_paper_dir, output_dict_dir)
