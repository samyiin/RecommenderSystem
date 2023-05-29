import os
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


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
    PAPER_EMBEDDING_COL = 'embedding'
    PAPER_ID_COL = 'corpusid'
    PAPER_TITLE_COL = 'title'
    PAPER_AUTHORS_COL = 'authors'
    PAPER_VENUE_COL = 'venue'

    # database queries
    COSINE_SIMILARITY = 'cosine similarity'  # use the string both as a query string and a column name
    GENERATE_DEFAULT_USERS = 'generate default users'
    GET_CITED_PAPERS = 'get cited papers'
    GET_BY_ID = "get row by id"

    def __init__(self, embedding_method, citation_db):
        """
        This function will read all the papers form the information source, and convert them into some kinds of
        representation. In the first edition, it will be an openAI 1536-dimension-vector.
        """
        self.citation_db = citation_db
        self.embedding_method = embedding_method
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
            papers_info_fp = os.path.join(info_source_dir, filename)
            if not os.path.isfile(papers_info_fp):
                raise Exception('this should not happen')
            # this part will change once I have a database
            output_fp = os.path.join(represented_items_dir, str(self.file_count))
            self._process_and_save_file_to_database(papers_info_fp, output_fp)
            self.file_count += 1
            self.data_fps.append(output_fp)

    def _process_and_save_file_to_database(self, papers_info_fp, output_fp):
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
                if temp_counter > 1000:
                    break
                temp_counter += 1
        papers_db = pd.DataFrame(papers)
        papers_db[self.PAPER_EMBEDDING_COL] = papers_db.apply(self.embed_paper, axis=1)
        # author id is string, because int might overflow, but larger storage
        papers_db[self.PAPER_ID_COL] = papers_db[self.PAPER_ID_COL].astype(str)
        # the reason to use pickle here is because we don't want information loss
        papers_db.to_pickle(output_fp)

    def embed_paper(self, row):
        # todo determine how to embed the papers? just concatenate the metadata and embed?
        content = ' '.join(str(value) for value in row)
        # embedding = self.embedding_method.embed_short_text(content)
        return np.random.rand(1, 1536)  # need this shape for cosine similarity

    ########################################################################################################################
    ########################################################################################################################
    ######################                                                                 #################################
    ######################                     Database Services                           #################################
    ######################                                                                 #################################
    ########################################################################################################################
    ########################################################################################################################
    def query_database(self, args):
        """args is list of arguments, the first argument is query, the rest is possibly parameters for the query"""
        query = args[0]
        if query == self.COSINE_SIMILARITY:
            '''
            if it's cosine similarity, then the second arg is the vector to calculate, and the third is the size of 
            returned result
            '''
            vector = args[1]  # todo input check?
            result_limit_size = args[2]
            return self._order_by_cosine_similarity(vector, result_limit_size)
        if query == self.GENERATE_DEFAULT_USERS:
            store_df_fp = args[1]
            self._generate_default_user(store_df_fp)
        if query == self.GET_CITED_PAPERS:
            citing_paper_id = args[1]
            return self._get_citation_of_paper(citing_paper_id)
        if args[0] == self.GET_BY_ID:
            citing_paper_id = args[1]
            return self._get_row_by_id(citing_paper_id)

    def _get_row_by_id(self, paper_id):
        """
        Once I have a database, id should have a index for fast rerieval
        This will return a 1 line pandas dataframe
        """
        # todo this will be optimize once we have a database
        # todo this will be a query
        for represented_items_fp in self.data_fps:
            represented_items_df = pd.read_pickle(represented_items_fp)
            # find the rows of this user id (supposed to be 1 or 0 row)
            rows = represented_items_df.loc[represented_items_df[self.PAPER_ID_COL] == paper_id]
            if len(rows) > 1:
                raise Exception('this should not happen')
            if len(rows) == 1:
                return rows
        raise Exception('this user id does not exit!')

    def _order_by_cosine_similarity(self, vector, limit_size):
        """
        given a vector with 1536 dim, order the item database by the cosine similarity of the embedding with this vector
        """
        all_selected_items = []
        for represented_items_fp in self.data_fps:
            represented_items_df = pd.read_pickle(represented_items_fp)
            represented_items_df[self.COSINE_SIMILARITY] = represented_items_df[self.PAPER_EMBEDDING_COL].apply(
                cosine_similarity,
                args=(vector,))
            selected_items = represented_items_df.sort_values(self.COSINE_SIMILARITY, ascending=False).head(limit_size)
            all_selected_items.append(selected_items)
        selected_items = pd.concat(all_selected_items).sort_values(self.COSINE_SIMILARITY, ascending=False).head(
            limit_size)
        # todo never tested multiple file, could go wrong
        return selected_items

    def _get_citation_of_paper(self, paper_id):
        result = self.citation_db.query_database([self.citation_db.GET_BY_ID, paper_id])
        cited_paper_list = result[self.citation_db.CITED_PAPER_COL].values.tolist()
        return cited_paper_list

    def _generate_default_user(self, store_df_fp):
        """
        For the stage where we haven't set up a database
        This function will directly generate structured user profile.
        """
        col = ['authorid', 'externalids', 'url', 'name', 'aliases', 'affiliations',
               'homepage', 'papercount', 'citationcount', 'hindex', 'updated',
               'liked paper ids', 'embedding']
        # author id (string) -> {"name": name(string), "paper": [paper id] , "paper embeddings": [paper embeddings]}
        map_author_paper = {}

        def add_to_map_author_paper(row):
            """
            the row (pandas series) is a paper, we will create a user that represents this author
             assume all the following exists
             """
            authors = row[self.PAPER_AUTHORS_COL]
            paper_id = str(row[self.PAPER_ID_COL])
            cited_papers = self._get_citation_of_paper(paper_id)
            paper_embedding = row[self.PAPER_EMBEDDING_COL]
            cited_paper_embeddings = []
            # add cited papers' embeddings
            for cited_paper_id in cited_papers:
                cited_paper_info = self._get_row_by_id(cited_paper_id)
                cited_paper_embeddings.append(cited_paper_info.iloc[0][self.PAPER_EMBEDDING_COL])
            for author_dic in authors:
                author_id = author_dic['authorId']
                author_name = author_dic['name']
                # add author information
                if author_id not in map_author_paper:
                    map_author_paper.update({author_id: {"name": author_name,
                                                         "liked_papers": [paper_id] + cited_papers,
                                                         "paper_embeddings": [
                                                                                 paper_embedding] + cited_paper_embeddings}})
                else:
                    map_author_paper[author_id]["liked_papers"].append(paper_id)
                    map_author_paper[author_id]["paper_embeddings"].append(paper_embedding)

        # iterating over database
        for represented_items_fp in self.data_fps:
            represented_items_df = pd.read_pickle(represented_items_fp)
            # creating user profiles for authors of these papers
            represented_items_df.apply(add_to_map_author_paper, axis=1)

        all_users = []
        for author_id in map_author_paper:
            # create a standard user profile
            user = {'authorid': author_id,
                    'externalids': None,
                    'url': None,
                    'name': map_author_paper[author_id]['name'],
                    'aliases': None,
                    'affiliations': None,
                    'homepage': None, 'papercount': None, 'citationcount': None, 'hindex': None, 'updated': None,
                    'liked paper ids': map_author_paper[author_id]['liked_papers'],
                    'embedding': np.average(np.array(map_author_paper[author_id]['paper_embeddings']), axis=0)
                    }
            all_users.append(user)
        all_users_df = pd.DataFrame.from_records(all_users)
        all_users_df.to_pickle(store_df_fp)
