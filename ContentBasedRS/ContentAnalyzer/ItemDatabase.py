import os
import pandas as pd
import numpy as np
import json
from ContentBasedRS.tools import OpenAIEmbedding

EMBEDDING_COL = 'embedding'
PAPER_ID_COL = 'corpusid'

class ItemDatabase:
    """
    NOTE: this class should be called only one time!
    This class will serve as a database for retrieve tables of data
    In the first edition, there will be two types of table: the represented items, and the user profiles. Maybe in the
    future there will be a database for feed back.
    Also in the first edition, we will use pandas instead of a real database, but this class will provide an interphase
    for query data from database
    This class only works on Aviv's semantic scholar data, or anything that's similar in format.
    """

    def __init__(self, info_source_dir, represented_items_dir, embedding_method):
        """
        This function will read all the papers form the information source, and convert them into some kinds of
        representation. In the first edition, it will be an openAI 1536-dimension-vector.
        """
        self.embedding_method = embedding_method
        self.represent_items_fps = []
        file_count = 0
        for filename in os.listdir(info_source_dir):
            if filename != '0-06':
                # todo for testing, because this computer can't run bigger files....
                continue
            # fool safe check
            papers_info_fp = os.path.join(info_source_dir, filename)
            if not os.path.isfile(papers_info_fp):
                raise Exception('this should not happen')
            # this part will change once I have a database
            output_fp = represented_items_dir + str(file_count)
            self._save_file_to_database(papers_info_fp, output_fp)
            file_count += 1
            self.represent_items_fps.append(output_fp)
        self.has_initialized = True

    def _save_file_to_database(self, papers_info_fp, output_fp):
        """
        take a paper file from the semantic scholar papers,
        convert to pickle file
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
        papers_db[EMBEDDING_COL] = papers_db.apply(self._embed_paper, axis=1)
        # the reason to use pickle here is because we don't want information loss
        papers_db.to_pickle(output_fp)

    def _embed_paper(self, row):
        # todo determine how to embed the papers? just concatenate the metadata and embed?

        content = ' '.join(str(value) for value in row)
        # embed this content
        return np.random.rand(1536, )

    def get_item_by_id(self, paper_id):
        """
        Once I have a database, id should have a index for fast rerieval
        """
        # todo this will be optimize once we have a database
        for represented_items_fp in self.represent_items_fps:
            represented_items_df = pd.read_pickle(represented_items_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            rows = represented_items_df.loc[represented_items_df[PAPER_ID_COL] == paper_id]
            if len(rows) > 1:
                raise Exception('this should not happen')
            if len(rows) == 1:
                return rows
        raise Exception('this user id does not exit!')
