"""This script is only for this database."""

import os
import sqlite3
import pandas as pd
import json
import sys
import numpy as np
import pickle


# ----------------------------------------helper functions--------------------------------------------------------------
def _info_to_BLOB(row):
    return pickle.dumps(row)


def _BLOB_to_info(row):
    return pickle.loads(row)


def for_each_row_do(conn, callback, args):
    """
    because there is buffer limit, so I build this function for code reuse
    the callback's first parameter is row
    todo what if there is no argument?
    if there is 1 argument, use (the_arg, ) tuple. else use tuple of arguments
    This function can only read lines from database, cannot write to database,
    else might trigger error like "database is lockd"
    """

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    map_author_paper = {}
    offset = 0
    buffer_size = BUFFER_SIZE

    while True:
        # read database paper table
        # read by buffer
        query = f"select * from paper limit {buffer_size} offset {offset}"

        # update offset
        offset += buffer_size

        # directly read into pandas
        papers_df = pd.read_sql_query(query, conn)
        if papers_df.empty:
            break
        papers_df[COL_EMBEDDING] = papers_df[COL_EMBEDDING].apply(_BLOB_to_info)
        papers_df[COL_AUTHORS] = papers_df[COL_AUTHORS].apply(_BLOB_to_info)
        papers_df.apply(callback, args=args, axis=1)

    conn.close()


# ----------------------------------------initialize functions----------------------------------------------------------
def create_paper_table(conn):
    """
    create a database named MY_DATABASE if not exist,
    create a table named TABLE_MAIN in the database
    """

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # if the table already exists then we will drop it. Caution: if accidentally call this function might delete
    # important details
    query = f'''DROP TABLE IF EXISTS {TABLE_MAIN};'''
    cursor.execute(query)

    # Define the SQL query to create the table
    query = '''
            CREATE TABLE paper (
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
    cursor.execute(query)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and the database connection
    cursor.close()
    conn.close()


# def order_by_cosine_similarity(my_database):
#     # Connect to the SQLite database(create one if not exist)
#     conn = sqlite3.connect(MY_DATABASE)
#
#     # Create a cursor object to execute SQL queries
#     cursor = conn.cursor()
#     map_author_paper = {}
#     offset = 0
#     buffer_size = BUFFER_SIZE
#
#     while True:
#         # read database paper table
#         # read by buffer
#         query = f"select * from paper limit {buffer_size} offset {offset}"
#
#         # update offset
#         offset += buffer_size
#
#         # directly read into pandas
#         papers_df = pd.read_sql_query(query, conn)
#         if papers_df.empty:
#             break
#         papers_df[COL_EMBEDDING] = papers_df[COL_EMBEDDING].apply(_BLOB_to_info)
#         papers_df[COL_AUTHORS] = papers_df[COL_AUTHORS].apply(_BLOB_to_info)
#         papers_df.apply(callback, args=args, axis=1)
#
#     conn.close()

# ---------------------------------------------test initialize----------------------------------------------------------
TABLE_MAIN = 'paper'

# column names for the main table
COL_PAPER_ID = 'paperId'
COL_CORPUS_ID = 'corpusId'
COL_TITLE = 'title'
COL_ABSTRACT = 'abstract'
COL_TLDR = 'tldr'  # tldr summary
COL_REF_COUNT = 'referenceCount'
COL_CITE_COUNT = 'citationCount'
COL_INFLUENTIAL_CITE_COUNT = 'influentialCitationCount'
COL_EMBEDDING = 'embedding'
COL_PUBLICATION_DATE = 'publicationDate'
COL_AUTHORS = 'authors'

ATTR_AUTHORS_ID = 'authorId'
ATTR_AUTHORS_NAME = 'name'

MY_DATABASE = '/cs/labs/avivz/hsiny/recommender_system/Content.db'

BUFFER_SIZE = 9


class ContentDB:
    def __init__(self):
        self.my_database = MY_DATABASE
        self.conn = sqlite3.connect(self.my_database)

    # todo should make a tool file
    def _give_paper_embedding(self, paper_df):
        attr_embedding_vec = 'vector'
        paper_df[COL_EMBEDDING] = paper_df[COL_EMBEDDING].apply(lambda row: row[attr_embedding_vec])
        return paper_df

    # -----------------------------------------------API functions------------------------------------------------------
    def add_papers_to_db(self, conn, raw_source_dir):
        for filename in os.listdir(raw_source_dir):
            with open(os.path.join(raw_source_dir, filename), 'r') as papers_file:
                data = json.load(papers_file)
            papers_df = pd.DataFrame.from_records(data)
            # process data -- potentially adding open ai embeddings
            papers_df = self._give_paper_embedding(papers_df)

            # convert arrays to binary large object because that what sqlite have...
            # Modern problem requires modern solution
            papers_df[COL_EMBEDDING] = papers_df[COL_EMBEDDING].apply(_info_to_BLOB)
            papers_df[COL_AUTHORS] = papers_df[COL_AUTHORS].apply(_info_to_BLOB)
            papers_df[COL_TLDR] = papers_df[COL_TLDR].apply(_info_to_BLOB)

            # output directly from pandas to database
            # todo try to append same paper id will fail.
            papers_df.to_sql('paper', conn, if_exists='append', index=False)
