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

MY_DATABASE = '/cs/labs/avivz/hsiny/recommender_system/Profile.db'

TABLE_MAIN = "author"

# column names of main table
COL_AUTHOR_ID = 'authorId'
COL_NAME = 'name'
COL_KNOWN_PAPERS = 'known_papers'
COL_EMBEDDING = 'embedding'  # array of shape (736, )

PAPER_KIND_WRITE = "write"
PAPER_KIND_CITE = 'cite'
PAPER_KINDS = [PAPER_KIND_WRITE, PAPER_KIND_CITE]


def _info_to_BLOB(row):
    return pickle.dumps(row)


def _BLOB_to_info(row):
    return pickle.loads(row)


# ----------------------------------------initialize functions----------------------------------------------------------

# open the user database, and read chunck by chunck to create profile database.
def create_author_table(conn):
    """create a table named 'paper' in my database"""

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    '''If we want to initialize database, we drop paper table. else will raise warning in case delete the table by
     mistake'''
    query = f'''DROP TABLE IF EXISTS {TABLE_MAIN};'''
    cursor.execute(query)

    # Define the SQL query to create the table
    query = f'''
            CREATE TABLE {TABLE_MAIN} (
            authorId TEXT PRIMARY KEY,
            name TEXT,
            known_papers BLOB,  
            embedding BLOB
            )
            '''
    # known papers will be dictionary of paper_kind -> set of paper_ids

    # Execute the SQL query to create the table
    cursor.execute(query)

    # todo create index for id for fast access, also for paper

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and the database connection
    cursor.close()
    conn.close()


# -----------------------------------------------API functions----------------------------------------------------------


def update_author(conn, authorId, name, paper_id, paper_kind, embedding):
    """
    update author with authorId, one paper at a time
    give author id, if the author in database, then update it, else insert into it
    """
    cursor = conn.cursor()
    author = pd.read_sql_query(f"select * from {TABLE_MAIN} where authorId == {authorId}", conn)
    if author.empty:
        # initialize an known paper dictionary
        known_papers = {}
        for kind in PAPER_KINDS:
            known_papers.update({kind: set()})
        known_papers[paper_kind].add(paper_id)
        new_row = (authorId, name, _info_to_BLOB(known_papers), _info_to_BLOB(embedding))
        query = f"insert into {TABLE_MAIN} {COL_AUTHOR_ID, COL_NAME, COL_KNOWN_PAPERS, COL_EMBEDDING} values (?, ?, ?, ?)"
        cursor.execute(query, new_row)

    else:
        # there should be only 1 row because authorId is a unique Key
        author_row = author.iloc[0]
        old_embedding = _BLOB_to_info(author_row[COL_EMBEDDING])
        known_papers = _BLOB_to_info(author_row[COL_KNOWN_PAPERS])
        # if the paper is already known, then nothing changes
        # paper kind must already in the dictionary
        if paper_id not in known_papers[paper_kind]:
            known_papers[paper_kind].add(paper_id)
            # update embedding todo this might change depends on how we calculate the embedding
            total_paper_number = sum([len(paper_set) for paper_set in known_papers.values()])
            new_embedding = ((total_paper_number - 1) * np.array(old_embedding) + np.array(
                embedding)) / total_paper_number
            update_info = (_info_to_BLOB(new_embedding), _info_to_BLOB(known_papers), authorId)
            query = f"update author set {COL_EMBEDDING}=?, {COL_KNOWN_PAPERS}=? where {COL_AUTHOR_ID}=?"
            cursor.execute(query, update_info)


