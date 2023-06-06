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
        self.COL_EMBEDDING = 'embedding'

        # column names of main table
        self.COL_IDX_AUTHOR_ID = 0
        self.COL_IDX_NAME = 1
        self.COL_IDX_KNOWN_PAPERS = 2
        self.COL_IDX_EMBEDDING = 3

        self.PAPER_KIND_WRITE = "write"
        self.PAPER_KIND_CITE = 'cite'
        self.PAPER_KINDS = [self.PAPER_KIND_WRITE, self.PAPER_KIND_CITE]

        self.conn = sqlite3.connect(self.my_database)
        self.cursor = self.conn.cursor()

    def commit_change(self):
        # Commit the changes to the database
        self.conn.commit()

    def open_connection(self):
        self.conn = sqlite3.connect(self.my_database)
        self.cursor = self.conn.cursor()

    def close_connection(self):
        # Close the cursor and the database connection
        self.cursor.close()
        self.conn.close()

    def query_database(self, query):
        """return a iterator of queried result, and the column number"""
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

        # Define the SQL query to create the table
        query = f'''
                CREATE TABLE {self.MAIN_TABLE_NAME} (
                authorId TEXT PRIMARY KEY,
                name TEXT,
                known_papers BLOB,  
                embedding BLOB
                )
                '''
        # known papers will be dictionary of paper_kind -> set of paper_ids

        # Execute the SQL query to create the table
        self.cursor.execute(query)

        # todo create index for id for fast access, also for paper

    # -----------------------------------------------API functions------------------------------------------------------

    def update_author(self, authorId, name, paper_id, paper_kind, embedding):
        """
        update author with authorId, one paper at a time
        give author id, if the author in database, then update it, else insert into it
        """
        author = pd.read_sql_query(f"select * from {self.MAIN_TABLE_NAME} where authorId == {authorId}", self.conn)
        if author.empty:
            # initialize an known paper dictionary
            known_papers = {}
            for kind in self.PAPER_KINDS:
                known_papers.update({kind: set()})
            known_papers[paper_kind].add(paper_id)
            new_row = (authorId, name, info_to_BLOB(known_papers), info_to_BLOB(embedding))
            query = f"insert into {self.MAIN_TABLE_NAME} {self.COL_AUTHOR_ID, self.COL_NAME, self.COL_KNOWN_PAPERS, self.COL_EMBEDDING} values (?, ?, ?, ?)"
            self.cursor.execute(query, new_row)

        else:
            # there should be only 1 row because authorId is a unique Key
            author_row = author.iloc[0]
            old_embedding = BLOB_to_info(author_row[self.COL_EMBEDDING])
            known_papers = BLOB_to_info(author_row[self.COL_KNOWN_PAPERS])
            # if the paper is already known, then nothing changes
            # paper kind must already in the dictionary
            if paper_id not in known_papers[paper_kind]:
                known_papers[paper_kind].add(paper_id)
                # update embedding todo this might change depends on how we calculate the embedding
                total_paper_number = sum([len(paper_set) for paper_set in known_papers.values()])
                new_embedding = ((total_paper_number - 1) * np.array(old_embedding) + np.array(
                    embedding)) / total_paper_number
                update_info = (info_to_BLOB(new_embedding), info_to_BLOB(known_papers), authorId)
                query = f"update author set {self.COL_EMBEDDING}=?, {self.COL_KNOWN_PAPERS}=? where {self.COL_AUTHOR_ID}=?"
                self.cursor.execute(query, update_info)
