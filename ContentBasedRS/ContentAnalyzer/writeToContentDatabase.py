import os
import sqlite3
import pandas as pd
import json
import sys
import numpy as np
import pickle

MY_DATABASE = '/cs/labs/avivz/hsiny/recommender_system/recommender_system.db'
RAW_PAPER_JSON = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RawSource/papers/sample_papers.json'


def embedding_to_numpy_pickles(row):
    return pickle.dumps(np.array(row['vector']))


def tldr_to_string(row):
    if row is None:
        return ''
    return row['text']


def authors_to_json(row):
    return json.dumps(row)

def add_papers_to_db():
    with open(RAW_PAPER_JSON, 'r') as papers_file:
        data = json.load(papers_file)
    papers_df = pd.DataFrame.from_records(data)
    # process data


    # convert numpy to bytes to not lose information. Modern problems need modern solutions
    papers_df['embedding'] = papers_df['embedding'].apply(embedding_to_numpy_pickles)
    papers_df['tldr'] = papers_df['tldr'].apply(tldr_to_string)
    papers_df['authors'] = papers_df['authors'].apply(authors_to_json)

    # connect to database
    conn = sqlite3.connect(MY_DATABASE)

    papers_df.to_sql('paper', conn, if_exists='replace', index=False)  # if exist can be append too

    # close the connection
    conn.close()


def create_paper_table():
    # Connect to the SQLite database
    conn = sqlite3.connect(MY_DATABASE)

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Define the SQL query to create the table
    query = '''
            CREATE TABLE paper (
            id INTEGER PRIMARY KEY,
            column1 TEXT,
            column2 INTEGER
            -- Other column definitions
            )
            '''

    # Execute the SQL query to create the table
    cursor.execute(query)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and the database connection
    cursor.close()
    conn.close()


