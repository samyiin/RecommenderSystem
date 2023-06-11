"""
initialization: This should only run once!
why don't I do it by database? because it's web request so we need parallel request

"""

from ContentBasedRS.ContentAnalyzer import *
from ContentBasedRS.ProfileLearner import *
from ContentBasedRS.FilteringComponent import *
from ContentBasedRS.Utils import *
import os
from pandarallel import pandarallel
import pandas as pd
import requests
from ContentBasedRS.Utils import *
from ContentBasedRS.ContentAnalyzer import *
from joblib import Memory


def initialize_paper_database():
    contentDB = ContentDB()
    contentDB.create_main_table()
    raw_paper_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RawSource/papers'
    contentDB.add_papers_to_db(raw_paper_dir)


def initialize_author_database():
    def _record_default_authors(row, args):
        """
        This is a callback for pandas dataframe.
        for each row: add the author to author table, if not exist, else, update the author
        todo: Gigantic room for speed improvement, not just realize the function
        """
        authors = BLOB_to_info(row[contentDB.get_col_index(contentDB.COL_AUTHORS)])
        wrote_paper_id = row[contentDB.get_col_index(contentDB.COL_PAPER_ID)]
        ref_paper = BLOB_to_info(row[contentDB.get_col_index(contentDB.COL_REFERENCE)])
        ref_paper_ids = set([paper_info_dic['paperId'] for paper_info_dic in ref_paper])
        known_papers = {profileDB.PAPER_KIND_WRITE: {wrote_paper_id},
                        profileDB.PAPER_KIND_REF: ref_paper_ids}
        for author_info in authors:
            author_name = author_info[contentDB.ATTR_AUTHORS_NAME]
            author_id = author_info[contentDB.ATTR_AUTHORS_ID]
            # todo what todo when there is no author id?
            if author_id is None:
                continue
            profileDB.update_author(author_id, author_name, known_papers)

    # # CAUTION: Running this will delete all data!
    profileDB.create_main_table()
    #
    # # record default authors for profile db
    contentDB.for_each_row_do(_record_default_authors, args=[])


def create_embedding_table():
    """Don't want to bother write modularity here, just uncomment TABLE_NAME to the table you want to initialize"""

    # for openAI embedding
    query = f'''DROP TABLE IF EXISTS {contentDB.OPENAI_EMBEDDING_TABLE_NAME};'''
    contentDB.cursor.execute(query)

    query = f'''
            CREATE TABLE {contentDB.OPENAI_EMBEDDING_TABLE_NAME} (
            {contentDB.COL_PAPER_ID} TEXT PRIMARY KEY,
            {contentDB.COL_EMBEDDING} BLOB
            )
            '''
    contentDB.cursor.execute(query)

    # for specter embedding
    query = f'''DROP TABLE IF EXISTS {contentDB.SPECTER_EMBEDDING_TABLE_NAME};'''
    contentDB.cursor.execute(query)

    query = f'''
                CREATE TABLE {contentDB.SPECTER_EMBEDDING_TABLE_NAME} (
                {contentDB.COL_PAPER_ID} TEXT PRIMARY KEY,
                {contentDB.COL_EMBEDDING} BLOB
                )
                '''
    contentDB.cursor.execute(query)


"""cache_function_call the result here so we don't need to request again..."""
cache_dir = 'cache_function_call/specter'
memory = Memory(cache_dir)


@memory.cache
def SPECTURE_EMBEDDING(row):
    """will return a one dimensional list"""
    # todo fake, set title to be id here for request from semantic shcolar
    fields = "embedding"
    # time.sleep(0.01)     # in case too much request
    paper_id = row[contentDB.COL_PAPER_ID]
    print(paper_id)
    rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/' + paper_id,
                       headers={'x-api-key': S2_API_KEY},
                       params={'fields': fields})
    rsp.raise_for_status()
    results = rsp.json()
    embedding = results['embedding']['vector']
    return info_to_BLOB(embedding)


cache_dir = 'cache_function_call/openAi'
memory = Memory(cache_dir)


@memory.cache
def OPENAI_EMBEDDING(row):
    title = row[contentDB.COL_TITLE]
    abstract = row[contentDB.COL_ABSTRACT]
    content = title
    if abstract is not None:
        content += abstract
    embedding = embed_long_text(content)
    return info_to_BLOB(embedding)


def apply_embedding_to_row():
    """
    Don't want to bother write modularity here, just uncomment and comment the code you want,
    OpenAI embedding can't run with SPECTER embedding, one is uncomment - the other should be commented
    """
    query = f"""
            select {contentDB.COL_PAPER_ID}, {contentDB.COL_TITLE}, {contentDB.COL_ABSTRACT}
            from {contentDB.MAIN_TABLE_NAME}
            limit 10000 offset 0 
            """
    papers_df = pd.read_sql(query, contentDB.conn)

    # for SPECTER embedding
    # papers_df[contentDB.COL_EMBEDDING] = papers_df.parallel_apply(SPECTURE_EMBEDDING, axis=1)
    # papers_df = papers_df[[contentDB.COL_PAPER_ID, contentDB.COL_EMBEDDING]]
    # papers_df.set_index(contentDB.COL_PAPER_ID, inplace=True)
    # papers_df.to_sql(contentDB.SPECTER_EMBEDDING_TABLE_NAME, contentDB.conn, if_exists='append')  # handling replicate

    # for OpenAI embedding can't run with SPECTER embedding
    papers_df[contentDB.COL_EMBEDDING] = papers_df.parallel_apply(OPENAI_EMBEDDING, axis=1)
    papers_df = papers_df[[contentDB.COL_PAPER_ID, contentDB.COL_EMBEDDING]]
    papers_df.set_index(contentDB.COL_PAPER_ID, inplace=True)
    papers_df.to_sql(contentDB.OPENAI_EMBEDDING_TABLE_NAME, contentDB.conn, if_exists='append')  # handling replicate

S2_API_KEY = os.getenv('S2_API_KEY')  # export S2_API_KEY=xxxxx (no space for equal sign)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
pandarallel.initialize()

# connect to databases
profileDB = ProfileDB()
contentDB = ContentDB()

initialize_paper_database()
contentDB.commit_change()
profileDB.commit_change()

initialize_author_database()
contentDB.commit_change()
profileDB.commit_change()

create_embedding_table()
contentDB.commit_change()
profileDB.commit_change()

apply_embedding_to_row()
contentDB.commit_change()
profileDB.commit_change()
