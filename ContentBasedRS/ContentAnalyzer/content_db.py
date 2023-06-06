"""This script is only for this database."""

import os
import sqlite3
import pandas as pd
import json
from ContentBasedRS.Utils import *



# ----------------------------------------helper functions--------------------------------------------------------------


class ContentDB:

    def __init__(self):
        self.MAIN_TABLE_NAME = 'paper'

        # column names for the main table
        self.COL_PAPER_ID = 'paperId'
        self.COL_CORPUS_ID = 'corpusId'
        self.COL_TITLE = 'title'
        self.COL_ABSTRACT = 'abstract'
        self.COL_TLDR = 'tldr'  # tldr summary
        self.COL_REF_COUNT = 'referenceCount'
        self.COL_CITE_COUNT = 'citationCount'
        self.COL_INFLUENTIAL_CITE_COUNT = 'influentialCitationCount'
        self.COL_EMBEDDING = 'embedding'
        self.COL_PUBLICATION_DATE = 'publicationDate'
        self.COL_AUTHORS = 'authors'

        self.COL_IDX_PAPER_ID = 0
        self.COL_IDX_CORPUS_ID = 1
        self.COL_IDX_TITLE = 2
        self.COL_IDX_ABSTRACT = 3
        self.COL_IDX_REF_COUNT = 4
        self.COL_IDX_CITE_COUNT = 5
        self.COL_IDX_INFLUENTIAL_CITE_COUNT = 6
        self.COL_IDX_EMBEDDING = 7
        self.COL_IDX_TLDR = 8
        self.COL_IDX_PUBLICATION_DATE = 9
        self.COL_IDX_AUTHORS = 10

        self.ATTR_AUTHORS_ID = 'authorId'
        self.ATTR_AUTHORS_NAME = 'name'

        self.BUFFER_SIZE = 9

        self.my_database = '/cs/labs/avivz/hsiny/recommender_system/Content.db'
        self.conn = sqlite3.connect(self.my_database)
        self.cursor = self.conn.cursor()
        # Close the cursor and the database connection
        # cursor.close()
        # conn.close()

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
    # todo should make a tool file
    def _give_paper_embedding(self, paper_df):
        attr_embedding_vec = 'vector'
        paper_df[self.COL_EMBEDDING] = paper_df[self.COL_EMBEDDING].apply(lambda row: row[attr_embedding_vec])
        return paper_df

    def create_main_table(self):
        """
        create a database named MY_DATABASE if not exist,
        create a table named TABLE_MAIN in the database
        """

        # if the table already exists then we will drop it. Caution: if accidentally call this function might delete
        # important details
        query = f'''DROP TABLE IF EXISTS {self.MAIN_TABLE_NAME};'''
        self.cursor.execute(query)

        # Define the SQL query to create the table
        query = f'''
                CREATE TABLE {self.MAIN_TABLE_NAME} (
                paperId TEXT PRIMARY KEY,
                corpusId INTEGER,
                title TEXT,
                abstract TEXT,
                referenceCount INTEGER,
                citationCount INTEGER,
                influentialCitationCount INTEGER,
                embedding BLOB,
                tldr BLOB,
                publicationDate TEXT,
                authors BLOB
                )
                '''
        # Execute the SQL query to create the table
        self.cursor.execute(query)

    # -----------------------------------------------API functions------------------------------------------------------
    def add_papers_to_db(self, raw_source_dir):
        for filename in os.listdir(raw_source_dir):
            with open(os.path.join(raw_source_dir, filename), 'r') as papers_file:
                data = json.load(papers_file)
            papers_df = pd.DataFrame.from_records(data)
            # process data -- potentially adding open ai embeddings
            papers_df = self._give_paper_embedding(papers_df)

            # convert arrays to binary large object because that what sqlite have...
            # Modern problem requires modern solution
            papers_df[self.COL_EMBEDDING] = papers_df[self.COL_EMBEDDING].apply(info_to_BLOB)
            papers_df[self.COL_AUTHORS] = papers_df[self.COL_AUTHORS].apply(info_to_BLOB)
            papers_df[self.COL_TLDR] = papers_df[self.COL_TLDR].apply(info_to_BLOB)
            # output directly from pandas to database
            # todo try to append same paper id will fail.
            papers_df.to_sql('paper', self.conn, if_exists='append', index=False)

    def for_each_row_do(self, callback, args):
        """
        the callback's first parameter is row
        the the call back's second parameter is a list of arguments, can be empty list
        """
        query = 'select * from paper'
        self.cursor.execute(query)
        record = self.cursor.fetchall()
        for row in record:
            callback(row, args)

# ---------------------------------------------test initialize----------------------------------------------------------
