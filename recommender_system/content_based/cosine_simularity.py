from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq

RECOMMENDING_PAPER_NUMBER = 10
EMBEDDING_COL = 'embedding'  # todo think about the dependencies
CONTENT_COL = 'content'
TITLE_COL = 'title'
SIMILARITY_COL = 'similarity'


def recommend(preference_list, paper_database):
    """

    :param preference_list: this is a list of preference(string) of user, can be what the user searched
    :param paper_database: this is a database of papers, we assume it is pandas now
    :return: a pandas df that have recommended papers
    """
    preference_embedding = convert_preference_to_embeddings(preference_list)
    to_recommend = query_database(preference_embedding, paper_database)

    return to_recommend


def convert_preference_to_embeddings(preference_list):
    return np.random.rand(1, 1536)  # use fake embedding for now, later change to openAI embedding


def query_database(preference_embedding, paper_database):
    """this method depends on how the database looks like, here we first assume it's a pandas dataframe"""
    paper_database[SIMILARITY_COL] = paper_database[EMBEDDING_COL].apply(cosine_similarity, args=(preference_embedding,))
    sorted_papers = paper_database.sort_values(SIMILARITY_COL)
    to_recommend = sorted_papers[:RECOMMENDING_PAPER_NUMBER]
    return to_recommend
