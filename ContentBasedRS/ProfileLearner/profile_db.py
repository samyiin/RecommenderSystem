"""
Requires an existing DB
This database(table) depends on ContentDB.
API:
create_author_table: create an empty author table in the database, erase old author table if exists.
add_default_authors: generate author profile from papers in ContentDB, and add to ProfileDB
update_author: add a new author(user) to ProfileDB
"""

import pickle
import sqlite3

import numpy as np
import pandas as df
import pandas as pd
from ContentBasedRS.ContentAnalyzer import *
from ContentBasedRS.Utils import *


class ProfileDB:
    def __init__(self):
        self.my_database = '/cs/labs/avivz/hsiny/recommender_system/Profile.db'

        self.MAIN_TABLE_NAME = "author"

        # column names of main table
        self.COL_AUTHOR_ID = 'authorId'
        self.COL_NAME = 'name'
        self.COL_KNOWN_PAPERS = 'known_papers'

        self.PAPER_KIND_WRITE = "write"
        self.PAPER_KIND_REF = 'reference'
        self.PAPER_KIND_LIKED = 'liked'

        self.columns = [self.COL_AUTHOR_ID, self.COL_NAME, self.COL_KNOWN_PAPERS]
        self.open_connection()

    def __del__(self):
        self.commit_change()
        self.close_connection()
    def get_col_index(self, col_name):
        """assume col_name is legit, else will raise ValueError"""
        return self.columns.index(col_name)
    def commit_change(self):
        # Commit the changes to the database
        self.conn.commit()
        print("committed changes")

    def open_connection(self):
        self.conn = sqlite3.connect(self.my_database)
        self.cursor = self.conn.cursor()
        print("established connection!")

    def close_connection(self):
        # Close the cursor and the database connection
        self.cursor.close()
        self.conn.close()
        print("closed connection")

    def query_database(self, query):
        """
        Note: if input is wrong, will throw exception, if query for empty set, will return empty list.
        return a iterator of queried result, and the column number
        """
        self.cursor.execute(query)
        return self.cursor.fetchall(), [column[0] for column in self.cursor.description]


    # ----------------------------------------initialize functions------------------------------------------------------
    # open the user database, and read chunck by chunck to create profile database.
    def create_main_table(self):
        """create a table named 'paper' in my database"""
        '''If we want to initialize database, we drop paper table. else will raise warning in case delete the table by
         mistake'''
        query = f'''DROP TABLE IF EXISTS {self.MAIN_TABLE_NAME};'''
        self.cursor.execute(query)
        print(f'deleted table {self.MAIN_TABLE_NAME}')

        # Define the SQL query to create the table
        query = f'''
                CREATE TABLE {self.MAIN_TABLE_NAME} (
                {self.COL_AUTHOR_ID} TEXT PRIMARY KEY,
                {self.COL_NAME} TEXT,
                {self.COL_KNOWN_PAPERS} BLOB
                -- known papers will be dict ionary of paper_kind -> set of paper_ids
                )
                '''

        # Execute the SQL query to create the table
        self.cursor.execute(query)

        # todo create index for id for fast access, also for paper

    # -----------------------------------------------API functions------------------------------------------------------

    def update_author(self, authorId, name, known_papers):
        """
        update author with authorId, one paper at a time
        give author id, if the author in database, then update it, else insert into it
        There could be new fields adding in
        """
        author = pd.read_sql_query(f"select * from {self.MAIN_TABLE_NAME} where authorId == {authorId}", self.conn)
        if author.empty:
            new_row = (authorId, name, info_to_BLOB(known_papers))
            query = f"insert into {self.MAIN_TABLE_NAME} {self.COL_AUTHOR_ID, self.COL_NAME, self.COL_KNOWN_PAPERS} values (?, ?, ?)"
            self.cursor.execute(query, new_row)

        else:
            # there should be only 1 row because authorId is a unique Key
            author_row = author.iloc[0]
            old_known_papers = BLOB_to_info(author_row[self.COL_KNOWN_PAPERS])
            # merge old known papers and given known_papers
            new_known_paper = {}
            for key in old_known_papers.keys() | known_papers.keys():
                new_known_paper[key] = old_known_papers.get(key, set()) | known_papers.get(key, set())
            update_info = (info_to_BLOB(new_known_paper), authorId)
            query = f"update author set {self.COL_KNOWN_PAPERS}=? where {self.COL_AUTHOR_ID}=?"
            self.cursor.execute(query, update_info)

    def get_author_embedding(self, known_papers, contentDB):
        """
        Known_papers represents the dictionary of known papers, write or referenced
        contentDB is the paper database where we request embeddings from
        calculate author's embedding in run time because
        1. the way of calculation might change,
        2. The embedding format might change,
        3. it's not that often to calculate an author's embedding
        for now assuming given each paper, write or referenced, the same weight when calculate the average of the papers
        embeddings. And the author's embedding is the average of papers embeddings
        """
        write_papers = known_papers[self.PAPER_KIND_WRITE]
        referenced_papers = known_papers[self.PAPER_KIND_REF]
        all_known_papers = write_papers | referenced_papers
        if self.PAPER_KIND_LIKED in known_papers:
            liked_papers = known_papers[self.PAPER_KIND_LIKED]
            all_known_papers |= liked_papers
        sum_embedding = None
        num_embedding = 0
        for paper_id in all_known_papers:
            query = f"select {contentDB.COL_EMBEDDING} from {contentDB.EMBEDDING_TABLE} where {contentDB.COL_PAPER_ID} = '{paper_id}'"
            result, _ = contentDB.query_database(query)
            if result:  # result could be empty list []
                paper_embedding = np.array(BLOB_to_info(result[0][0]))
                num_embedding += 1
                if sum_embedding is None:
                    sum_embedding = paper_embedding
                else:

                    sum_embedding += paper_embedding
            else:
                # todo how to handle if paper is not in database?
                print(f"paper {paper_id} doesn't exist in database")
        author_embedding = sum_embedding / num_embedding
        return author_embedding

    def clear_liked_papers(self, authorId):
        author = pd.read_sql_query(f"select * from {self.MAIN_TABLE_NAME} where authorId == {authorId}", self.conn)
        if author.empty:
            return
        else:
            # there should be only 1 row because authorId is a unique Key
            author_row = author.iloc[0]
            known_papers = BLOB_to_info(author_row[self.COL_KNOWN_PAPERS])
            # merge old known papers and given known_papers
            known_papers[self.PAPER_KIND_LIKED] = set()
            update_info = (info_to_BLOB(known_papers), authorId)
            query = f"update author set {self.COL_KNOWN_PAPERS}=? where {self.COL_AUTHOR_ID}=?"
            self.cursor.execute(query, update_info)


