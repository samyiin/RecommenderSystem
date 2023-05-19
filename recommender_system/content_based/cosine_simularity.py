from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq
from database.database_pandas import *
import embeding_methods
# todo this way of import is bad, import package not script

class RSCosSimilarity:
    def __init__(self):
        self.database = DBPandas()
        self.RECOMMENDING_PAPER_NUMBER = 10

    def recommend(self, preference_list):
        """
        :param preference_list: this is a list of preference(string) of user, can be what the user searched
        :param paper_database: this is a database of papers, we assume it is pandas now
        :return: a pandas df that have recommended papers
        """
        preference_embedding = self.convert_preference_to_embeddings(preference_list)
        to_recommend = self.database.query_by_cosine_similarity(preference_embedding)

        return to_recommend[:self.RECOMMENDING_PAPER_NUMBER]

    def convert_preference_to_embeddings(self, preference_list):
        embedder = embeding_methods.OpenAIEmbedder()
        return np.random.rand(1, 1536)  # use fake embedding for now, later change to openAI embedding








