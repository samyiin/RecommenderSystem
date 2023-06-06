import os
from ContentBasedRS_toy.tools import OpenAIEmbedding
import pandas as pd
import numpy as np
import json
from ContentBasedRS_toy.ContentAnalyzer import ItemDatabase

"""
User must have the following attributes
user = {'authorid': None,
        'externalids': None,
        'url': None,
        'name': None,
        'aliases':None,
        'affiliations':None,
        'homepage': None, 'papercount': None, 'citationcount': None, 'hindex': None, 'updated': None,
        'liked paper ids':None,
        'embedding': None
            }
"""


class ProfileDatabase:
    """
    This class will serve as a database for retrieve tables of data
    In the first edition, there will be two types of table: the represented items, and the user profiles. Maybe in the
    future there will be a database for feed back.
    Also in the first edition, we will use pandas instead of a real database, but this class will provide an interphase
    for query data from database
    This class only works on Aviv's semantic scholar data, or anything that's similar in format.
    """
    EMBEDDING_COL = 'embedding'
    LIKED_PAPER_IDS_COL = 'liked paper ids'
    USER_ID_COL = 'authorid'

    # database query
    GET_BY_ID = "get row by id"
    def __init__(self, embedding_method, item_db, store_data_dir):
        """
        This function will read all the papers form the information source, and convert them into some kinds of
        representation. In the first edition, it will be an openAI 1536-dimension-vector.
        """
        self.embedding_method = embedding_method
        self.store_data_dir = store_data_dir
        self.item_db = item_db
        self.file_count = 0
        self.item_db.query_database([item_db.GENERATE_DEFAULT_USERS, self.store_data_dir + "default users"])
        self.data_fps = [self.store_data_dir + "default users"]  # todo this is a temporary way of iterating all the database


########################################################################################################################
########################################################################################################################
######################                                                                 #################################
######################                      Modify Database                            #################################
######################                                                                 #################################
########################################################################################################################
########################################################################################################################
    def add_by_raw_info(self, info_source_dir):
        """add entities to database given directory of raw source files"""
        for filename in os.listdir(info_source_dir):
            papers_info_fp = os.path.join(info_source_dir, filename)
            if not os.path.isfile(papers_info_fp):
                raise Exception('this should not happen')
            # this part will change once I have a database
            output_fp = os.path.join(self.store_data_dir, str(self.file_count))
            self._process_and_save_file_to_database(papers_info_fp, output_fp)
            self.file_count += 1
            self.data_fps.append(output_fp)

    def _process_and_save_file_to_database(self, author_info_fp, output_fp):
        """
        take a paper file from the semantic scholar papers,
        convert to pickle file
        """
        users = []
        temp_counter = 0
        with open(author_info_fp, 'r') as file:
            for line in file:  # each line seems to be a dictionary
                content = json.loads(line)
                users.append(content)
                # todo for testing: how to solve the scalability problem? database, index by id?
                if temp_counter > 100:
                    break
                temp_counter += 1
        users_db = pd.DataFrame(users)
        # author id is string, because int might overflow
        users_db[self.USER_ID_COL] = users_db[self.USER_ID_COL].astype(str)
        users_db[self.LIKED_PAPER_IDS_COL] = users_db.apply(self._get_liked_papers, axis=1)
        users_db[self.EMBEDDING_COL] = users_db.apply(self.embed_user, axis=1)
        # this part will change once I have a database
        users_db.to_pickle(output_fp)

    def embed_user(self, row):
        """
        In the first edition, I will find all the papers this user cited or wrote, and find these papers' embeddings,
        and take unweighted average of these embeddings
        :param row:
        :return:
        """
        liked_paper_ids = row[self.LIKED_PAPER_IDS_COL]
        liked_paper_embeddings = []
        for paper_id in liked_paper_ids:
            paper = self.item_db.query_database([self.item_db.GET_BY_ID, paper_id])
            paper_embedding = paper.iloc[0][self.EMBEDDING_COL]
            liked_paper_embeddings.append(paper_embedding)
        user_embedding = np.average(np.array(liked_paper_embeddings), axis=0)
        # embed this content
        return user_embedding

    def _get_liked_papers(self, row):
        """return list of paper "corpus id": all the papers this author wrote, plus all the papers this author cited"""
        # todo find all the papers this author wrote and cited
        # from author -> all things author wrote -> paper file
        # from paper to all other papers this paper cited -> citation file
        # maybe we can start from someone who's work in in paper db, who's name is in author db, and we can find
        # citation file for that paper...
        user_id = row[self.USER_ID_COL]
        sample_liked_papers = ['58033818']
        cited_papers = self.item_db.query_database([self.item_db.GET_CITED_PAPERS, '58033818'])
        sample_liked_papers += cited_papers
        return np.array(sample_liked_papers)

    def query_database(self, args):
        """args is list of arguments, the first argument is query, the rest is possibly parameters for the query"""
        if args[0] == self.GET_BY_ID:
            citing_paper_id = args[1]
            return self._get_row_by_id(citing_paper_id)

    def _get_row_by_id(self, user_id):
        """
        Once I have a database, id should have a index for fast rerieval
        """
        # todo this will be optimize once we have a database
        # todo this will be a query
        for user_profiles_fp in self.data_fps:
            user_profile_df = pd.read_pickle(user_profiles_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            rows = user_profile_df.loc[user_profile_df[self.USER_ID_COL] == user_id]
            if len(rows) > 1:
                raise Exception('this should not happen')
            if len(rows) == 1:
                return rows
        raise Exception('this user id does not exit!')

    def update_user(self, row):
        """get the user feed back, and we will update the user"""
        # todo input check
        for user_profiles_fp in self.data_fps:
            user_profile_df = pd.read_pickle(user_profiles_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            user_index = user_profile_df.loc[user_profile_df[self.USER_ID_COL] == row.iloc[0][self.USER_ID_COL]].index
            # todo what if it doesn't exist?
            user_profile_df.loc[user_index] = row
            user_profile_df.to_pickle(user_profiles_fp)
            return
        raise Exception('this user id does not exit!')
