import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ContentBasedRS.Utils import *




class Recommender:
    def __init__(self, contentDB, profileDB):
        self.contentDB = contentDB
        self.profileDB = profileDB
        self.recommend_number = 10



    def vector_similarity(self, embedding_vector):
        def COSINE_SIMILARITY(vector_binary):
            vector = BLOB_to_info(vector_binary)
            similarity = cosine_similarity([vector], [embedding_vector])
            return similarity[0][0]
        self.contentDB.conn.create_function("COSINE_SIMILARITY", 1, COSINE_SIMILARITY)
        query = f"select *, COSINE_SIMILARITY({self.contentDB.COL_EMBEDDING}) as similarity " +\
                f"from {self.contentDB.MAIN_TABLE_NAME} " +\
                f"order by similarity desc "
        result, columns = self.contentDB.query_database(query)
        return pd.DataFrame(result[:self.recommend_number], columns=columns)

    def order_by_number(self, column_name, ascending=False):
        """assume the column is numerical, and we order by either ascending or descending"""
        order = 'desc'
        if ascending:
            order = 'asc'
        query = f"select * from {self.contentDB.MAIN_TABLE_NAME} order by {column_name} {order} "
        result, columns = self.contentDB.query_database(query)
        return pd.DataFrame(result[:self.recommend_number], columns=columns)

    def weighted_linear_combination_ranking(self, embedding_vector):
        def COSINE_SIMILARITY(vector_binary):
            vector = BLOB_to_info(vector_binary)
            similarity = cosine_similarity([vector], [embedding_vector])
            return similarity[0][0]
        def LINEAR_COMBINATION_SCORE(cos_sim_rank, REF_COUNT_rank, CITE_COUNT_rank, INFLUENTIAL_CITE_COUNT_rank):
            """assume all these rank is int"""
            final_score = 0.7*cos_sim_rank + 0.1*REF_COUNT_rank + 0.1*CITE_COUNT_rank + 0.1*INFLUENTIAL_CITE_COUNT_rank
            return final_score
        self.contentDB.conn.create_function("COSINE_SIMILARITY", 1, COSINE_SIMILARITY)
        self.contentDB.conn.create_function('LINEAR_COMBINATION_SCORE', 4, LINEAR_COMBINATION_SCORE)

        # assume all the criteria is the higher the better - except for ranking, the lower the better
        query = "select *, LINEAR_COMBINATION_SCORE(cos_rank, ref_rank, cite_rank, inf_cite_rank) as rank " +\
                "from(" +\
                "select *, "+\
                    "row_number() over (order by similarity desc) cos_rank," +\
                    f"row_number() over (order by {self.contentDB.COL_REF_COUNT} desc) ref_rank," +\
                    f"row_number() over (order by {self.contentDB.COL_CITE_COUNT} desc) cite_rank," +\
                    f"row_number() over (order by {self.contentDB.COL_INFLUENTIAL_CITE_COUNT} desc) inf_cite_rank " +\
                f"from (select *, COSINE_SIMILARITY({self.contentDB.COL_EMBEDDING}) as similarity from {self.contentDB.MAIN_TABLE_NAME}))" +\
                "order by rank asc"
        result, columns = self.contentDB.query_database(query)
        return pd.DataFrame(result[:self.recommend_number], columns=columns)



