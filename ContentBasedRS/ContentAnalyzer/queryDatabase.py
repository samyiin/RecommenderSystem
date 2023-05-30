import sqlite3
import pandas as pd
import pickle


def numpy_pickles_to_embedding(row):
    return pickle.loads(row)


def cosine_similarity(vector):
    # read_from database
    conn = sqlite3.connect('/cs/labs/avivz/hsiny/recommender_system/recommender_system.db')
    query = "select * from paper"
    papers_df = pd.read_sql_query(query, conn)
    papers_df['embedding'] = papers_df['embedding'].apply(numpy_pickles_to_embedding)
    conn.close()
