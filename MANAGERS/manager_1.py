"""
This manager will read semantic scholar's data, and do recommend.
"""
import json
import pandas as pd
import numpy as np
from recommender_system.content_based.cosine_simularity import *


# setup toy database: Not responsibility of manager!
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
paper_database = papers_db
# todo setup a real database for larger storage and faster retrieve, change recommend accordingly.
# todo change recommender system to class design
# todo change embedding to real embedding, change openAI's embedding method: based on all attribute?


recommended_papers = recommend(['history', 'law'], paper_database)
print(recommended_papers[['title', 'similarity']])

