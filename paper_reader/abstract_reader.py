from abc import ABC, abstractmethod
from typing import Dict


class PaperReader(ABC):
    @abstractmethod
    def parse(self, paper_fp) -> Dict:
        pass
