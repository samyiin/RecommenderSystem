import os
import sqlite3
import pandas as pd
import json
import sys
import numpy as np
import pickle

def embedding_to_numpy_pickles(row):
    return pickle.dumps(np.array(row['vector']))

def numpy_pickles_to_embedding(row):
    return pickle.loads(row)

def tldr_to_string(row):
    if row is None:
        return ''
    return row['text']
def authors_to_json(row):
    return json.dumps(row)

def numpy_to_BLOB(arr):
    return arr.tobytes()


with open('/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RawSource/papers/sample_papers.json',
          'r') as papers_file:
    data = json.load(papers_file)
papers_df = pd.DataFrame.from_records(data)
# convert numpy to bytes to not lose information. Modern problems need modern solutions
papers_df['embedding'] = papers_df['embedding'].apply(embedding_to_numpy_pickles)
papers_df['tldr'] = papers_df['tldr'] .apply(tldr_to_string)
papers_df['authors'] = papers_df['authors'].apply(authors_to_json)

# connect to database
conn = sqlite3.connect('/cs/labs/avivz/hsiny/recommender_system/recommender_system.db')

papers_df.to_sql('paper', conn, if_exists='replace', index=False)

# close the connection
conn.close()

# df = pd.read_json("ContentBasedRS/RawSource/papers/sample_papers.json", orient="records")
print("debug stop point")

# read_from database
conn = sqlite3.connect('/cs/labs/avivz/hsiny/recommender_system/recommender_system.db')
query = "select * from paper"
papers_df = pd.read_sql_query(query, conn)
papers_df['embedding'] = papers_df['embedding'].apply(numpy_pickles_to_embedding)
conn.close()
