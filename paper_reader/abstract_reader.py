from abc import ABC, abstractmethod
from typing import Dict
import re


class PaperReader(ABC):
    @abstractmethod
    def parse(self, paper_fp) -> Dict:
        """this function takes a file path to a paper pdf, and parse it into attribute dictionary"""
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
