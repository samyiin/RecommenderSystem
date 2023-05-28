import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


class CitationDatabase:
    """
    NOTE: this class should be called only one time!
    This class will serve as a database for retrieve tables of data
    In the first edition, there will be two types of table: the represented items, and the user profiles. Maybe in the
    future there will be a database for feed back.
    Also in the first edition, we will use pandas instead of a real database, but this class will provide an interphase
    for query data from database
    This class only works on Aviv's semantic scholar data, or anything that's similar in format.
    """
    CITING_PAPER_COL = 'citingcorpusid'
    CITED_PAPER_COL = 'citedcorpusid'

    # queries
    GET_BY_ID = "get row by id"
    def __init__(self):
        """
        This function will read all the citations form the information source, and convert them into some kinds of
        representation. In the first edition, it will be an openAI 1536-dimension-vector.
        """
        self.data_fps = []
        self.file_count = 0

########################################################################################################################
########################################################################################################################
######################                                                                 #################################
######################                      Modify Database                            #################################
######################                                                                 #################################
########################################################################################################################
########################################################################################################################
    def add_raw_info(self, info_source_dir, represented_items_dir):
        """add entities to database given directory of raw source files"""
        for filename in os.listdir(info_source_dir):
            # fool safe check
            citations_info_fp = os.path.join(info_source_dir, filename)
            if not os.path.isfile(citations_info_fp):
                raise Exception('this should not happen')
            # this part will change once I have a database
            output_fp = os.path.join(represented_items_dir, str(self.file_count))
            self._process_and_save_file_to_database(citations_info_fp, output_fp)
            self.file_count += 1
            self.data_fps.append(output_fp)

    def _process_and_save_file_to_database(self, citations_info_fp, output_fp):
        """
        take a citation file from the semantic scholar citations,
        convert to pickle file
        """
        citations = []
        temp_counter = 0
        with open(citations_info_fp, 'r') as file:
            for line in file:  # each line seems to be a dictionary
                content = json.loads(line)
                citations.append(content)
                # todo for testing: how to solve the scalability problem? database, index by id?
                if temp_counter > 1000:
                    break
                temp_counter += 1
        citations_db = pd.DataFrame(citations)
        # the reason to use pickle here is because we don't want information loss
        citations_db.to_pickle(output_fp)





########################################################################################################################
########################################################################################################################
######################                                                                 #################################
######################                     Database Services                           #################################
######################                                                                 #################################
########################################################################################################################
########################################################################################################################
    def query_database(self, args):
        """args is list of arguments, the first argument is query, the rest is possibly parameters for the query"""
        if args[0] == self.GET_BY_ID:
            citing_paper_id = args[1]
            return self._get_row_by_id(citing_paper_id)

    def _get_row_by_id(self, citing_paper_id):
        """
        get all the paper cited by the citing paper
        """
        # todo this will be optimize once we have a database
        for represented_items_fp in self.data_fps:
            represented_items_df = pd.read_pickle(represented_items_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            rows = represented_items_df.loc[represented_items_df[self.CITING_PAPER_COL] == citing_paper_id]
            return rows
        raise Exception('this user id does not exit!')

