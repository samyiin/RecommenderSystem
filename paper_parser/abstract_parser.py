from abc import ABC, abstractmethod
from typing import Dict
import re
import json


class PaperReader(ABC):
    @abstractmethod
    def parse_dir(self, raw_paper_dir, output_dict_dir) -> None:
        """
        this function takes a path to a directory that stores all the paper pdf, and parse it into attribute
        dictionary, and save them under the output_dict_dir
        return None
        """
        pass

    def trim_special_characters(self, attribute_dic: Dict):
        """this method will remove all the special characters"""
        for key in attribute_dic:
            value = attribute_dic[key]
            # pattern = r'[^\w\s]'
            # attribute_dic[key] = re.sub(pattern, '', value)
            value = value.replace("\n", " ")
            attribute_dic[key] = value
        return attribute_dic

    def write_attribute_dict_to_file(self, attribute_dict, file_path):
        with open(file_path, 'w') as file:
            json.dump(attribute_dict, file)

    def read_attribute_dict_from_file(self, file_path):
        with open(file_path, 'rb') as file:
            loaded_data = json.loads(file.read())
        return loaded_data
