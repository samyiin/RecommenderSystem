import os
from ContentBasedRS.tools import OpenAIEmbedding
import pandas as pd
import numpy as np
import json
from ContentBasedRS.ContentAnalyzer import ItemDatabase


EMBEDDING_COL = 'embedding'
LIKED_PAPER_IDS_COL = 'liked paper ids'
USER_ID_COL = 'authorid'


class ProfileDatabase:
    """
    This class will serve as a database for retrieve tables of data
    In the first edition, there will be two types of table: the represented items, and the user profiles. Maybe in the
    future there will be a database for feed back.
    Also in the first edition, we will use pandas instead of a real database, but this class will provide an interphase
    for query data from database
    This class only works on Aviv's semantic scholar data, or anything that's similar in format.
    """
    has_initialized = False  # in case there is two database objects

    def __init__(self, info_source_dir, user_profiles_dir, item_db, embedding_method):
        """
        This function will read all the papers form the information source, and convert them into some kinds of
        representation. In the first edition, it will be an openAI 1536-dimension-vector.
        """
        self.embedding_method = embedding_method
        self.item_db = item_db
        self.user_profiles_fps = []
        if self.has_initialized:
            return
        file_count = 0
        for filename in os.listdir(info_source_dir):
            if filename != '20230127_081509_00053_ss3hj_1c1985a6-6832-4f1f-974e-fdc608212843':
                # todo for testing, because this computer can't run bigger files.... scalability problem
                continue
            papers_info_fp = os.path.join(info_source_dir, filename)
            if not os.path.isfile(papers_info_fp):
                raise Exception('this should not happen')
            # this part will change once I have a database
            output_fp = user_profiles_dir + str(file_count)
            self._save_file_to_database(papers_info_fp, output_fp)
            file_count += 1
            self.user_profiles_fps.append(output_fp)
        self.has_initialized = True


    def _save_file_to_database(self, author_info_fp, output_fp):
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
        # todo when user gets bigger, user id might overflow
        users_db[USER_ID_COL] = users_db[USER_ID_COL].astype('int64')
        users_db[LIKED_PAPER_IDS_COL] = users_db.apply(self._get_liked_papers, axis=1)
        users_db[EMBEDDING_COL] = users_db.apply(self._embed_user, axis=1)
        # this part will change once I have a database
        users_db.to_pickle(output_fp)

    def _embed_user(self, row):
        """
        In the first edition, I will find all the papers this user cited or wrote, and find these papers' embeddings,
        and take unweighted average of these embeddings
        :param row:
        :return:
        """
        liked_paper_ids = row[LIKED_PAPER_IDS_COL]
        liked_paper_embeddings = []
        for paper_id in liked_paper_ids:
            paper = self.item_db.get_item_by_id(paper_id)
            paper_embedding = paper.iloc[0][EMBEDDING_COL]
            liked_paper_embeddings.append(paper_embedding)
        user_embedding = np.average(np.array(liked_paper_embeddings), axis=0)
        # embed this content
        return user_embedding

    def _get_liked_papers(self, row):
        """return list of paper "corpus id": all the papers this author wrote, plus all the papers this author cited"""
        # todo find all the papers this author wrote and cited
        # this might not be solved if we don't have all the database....
        # from author -> all things author wrote -> paper file
        # from paper to all other papers this paper cited -> citation file
        # maybe we can start from someone who's work in in paper db, who's name is in author db, and we can find
        # citation file for that paper...
        user_id = row[USER_ID_COL]
        sample_liked_papers = [58033818, 57183173, 147654154, 147887750]
        return np.array(sample_liked_papers)


    def get_profile_by_id(self, user_id):
        """
        Once I have a database, id should have a index for fast rerieval
        """
        # todo this will be optimize once we have a database
        for user_profiles_fp in self.user_profiles_fps:
            user_profile_df = pd.read_pickle(user_profiles_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            rows = user_profile_df.loc[user_profile_df[USER_ID_COL] == user_id]
            if len(rows) > 1:
                raise Exception('this should not happen')
            if len(rows) == 1:
                return rows
        raise Exception('this user id does not exit!')






