import json
import pandas as pd

def read_info(papers_info_fp):
    """
    for experiment
    """
    papers = []
    temp_counter = 0
    with open(papers_info_fp, 'r') as file:
        for line in file:  # each line seems to be a dictionary
            content = json.loads(line)
            papers.append(content)
            # todo for testing: how to solve the scalability problem? database, index by id?
            if temp_counter > 100:
                break
            temp_counter += 1
    papers_db = pd.DataFrame(papers)
    print(1)

papers_info_fp = '/cs/labs/avivz/avivz/semantic_scholar_data/citations/20230127_081540_00122_jcyz3_1d7c3ad8-0f23-40d8-b5a1-e85775a69c86'
read_info(papers_info_fp)

