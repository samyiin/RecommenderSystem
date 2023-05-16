import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_COL = 'embedding'
CONTENT_COL = 'content'
TITLE_COL = 'title'
SIMILARITY_COL = 'similarity'


class DBPandas:

    def __init__(self):
        sample_paper_file = '/cs/labs/avivz/avivz/semantic_scholar_data/papers/0-06'  # this file have about 30k papers
        paper_database = pd.DataFrame()  # use a pandas to simulate a database for now
        papers = []
        with open(sample_paper_file, 'r') as file:
            for line in file:  # each line seems to be a dictionary
                content = json.loads(line)
                papers.append(content)
        papers_db = pd.DataFrame(papers)  # use a pandas df to act as a database for now.
        papers_db[EMBEDDING_COL] = [np.random.rand(1, 1536) for _ in range(len(papers_db.index))]
        papers_db[CONTENT_COL] = ''
        self.paper_database = papers_db

    def query_by_cosine_similarity(self, vec):
        """this function should return a iterator that have all the paper's cosine similarity to the given vector"""
        self.paper_database[SIMILARITY_COL] = self.paper_database[EMBEDDING_COL].apply(cosine_similarity,
                                                                             args=(vec,))
        sorted_papers = self.paper_database.sort_values(SIMILARITY_COL)
        return sorted_papers

