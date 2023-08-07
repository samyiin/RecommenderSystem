import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ContentBasedRS.Utils import *


class Recommender:
    def __init__(self, contentDB, profileDB, recommend_number=10):
        self.contentDB = contentDB
        self.profileDB = profileDB
        self.recommend_number = recommend_number

    def _sort_by_vector_similarity(self, embedding_vector, exclude_known_papers):
        def COSINE_SIMILARITY(vector_binary):
            vector = BLOB_to_info(vector_binary)
            similarity = cosine_similarity([vector], [embedding_vector])
            return similarity[0][0]

        self.contentDB.conn.create_function("COSINE_SIMILARITY", 1, COSINE_SIMILARITY)
        modified_exclude_list = ["\"" + paper_id + "\"" for paper_id in exclude_known_papers]
        query = f"""
                select {self.contentDB.MAIN_TABLE_NAME}.*,
                        COSINE_SIMILARITY({self.contentDB.COL_EMBEDDING}) as similarity 
                from ({self.contentDB.MAIN_TABLE_NAME} 
                        inner join {self.contentDB.EMBEDDING_TABLE} 
                        on {self.contentDB.MAIN_TABLE_NAME}.{self.contentDB.COL_PAPER_ID} = {self.contentDB.EMBEDDING_TABLE}.{self.contentDB.COL_PAPER_ID}
                       ) 
                where {self.contentDB.MAIN_TABLE_NAME}.{self.contentDB.COL_PAPER_ID} not in ({','.join(modified_exclude_list)})
                order by similarity desc
                limit {self.recommend_number}
                """
        final_result = self.contentDB.query_database(query)
        return final_result

    def order_by_number(self, column_name, ascending=False):
        """assume the column is numerical, and we order by either ascending or descending"""
        order = 'desc'
        if ascending:
            order = 'asc'
        query = f"""select * from {self.contentDB.MAIN_TABLE_NAME} 
                    order by {column_name} {order} 
                    limit {self.recommend_number}
                """
        final_result = self.contentDB.query_database(query)
        return final_result

    def weighted_linear_combination_ranking(self, embedding_vector, exclude_known_papers,
                                            cos_weight, ref_weight, cite_weight, influence_weight, year_weight):
        # assume the four weights sum up to 1
        def COSINE_SIMILARITY(vector_binary):
            vector = BLOB_to_info(vector_binary)
            similarity = cosine_similarity([vector], [embedding_vector])
            return similarity[0][0]

        def LINEAR_COMBINATION_SCORE(cos_sim_rank, REF_COUNT_rank, CITE_COUNT_rank, INFLUENTIAL_CITE_COUNT_rank, YEAR_rank):
            """assume all these rank is int"""
            # final_score = 0.7 * cos_sim_rank + 0.1 * REF_COUNT_rank + 0.1 * CITE_COUNT_rank + 0.1 * INFLUENTIAL_CITE_COUNT_rank
            final_score = cos_weight * cos_sim_rank + ref_weight * REF_COUNT_rank + \
                          cite_weight * CITE_COUNT_rank + influence_weight * INFLUENTIAL_CITE_COUNT_rank+ year_weight * YEAR_rank

            return final_score

        self.contentDB.conn.create_function("COSINE_SIMILARITY", 1, COSINE_SIMILARITY)
        self.contentDB.conn.create_function('LINEAR_COMBINATION_SCORE', 5, LINEAR_COMBINATION_SCORE)
        modified_exclude_list = ["\"" + paper_id + "\"" for paper_id in exclude_known_papers]

        # assume all the criteria is the higher the better - except for ranking, the lower the better
        query = f"""
                select *, LINEAR_COMBINATION_SCORE(cos_rank, ref_rank, cite_rank, inf_cite_rank, year_rank) as rank
                from(
                    select *, 
                    row_number() over (order by similarity desc) cos_rank,
                    row_number() over (order by {self.contentDB.COL_REF_COUNT} desc) ref_rank,
                    row_number() over (order by {self.contentDB.COL_CITE_COUNT} desc) cite_rank,
                    row_number() over (order by {self.contentDB.COL_INFLUENTIAL_CITE_COUNT} desc) inf_cite_rank,
                    row_number() over (order by {self.contentDB.COL_YEAR} desc) year_rank
                    from (
                        select {self.contentDB.MAIN_TABLE_NAME}.*,
                                COSINE_SIMILARITY({self.contentDB.COL_EMBEDDING}) as similarity 
                        from ({self.contentDB.MAIN_TABLE_NAME} 
                                inner join {self.contentDB.EMBEDDING_TABLE} 
                                on {self.contentDB.MAIN_TABLE_NAME}.{self.contentDB.COL_PAPER_ID} = {self.contentDB.EMBEDDING_TABLE}.{self.contentDB.COL_PAPER_ID}
                               ) 
                        )
                    where {self.contentDB.COL_PAPER_ID} not in ({','.join(modified_exclude_list)})
                    )
                order by rank asc
                limit {self.recommend_number}
                """
        final_result = self.contentDB.query_database(query)
        return final_result

    # ----------------------------------------higher level functionalities----------------------------------------------
    def search_engine(self, keyword):
        keyword_embedding = OpenAIEmbedding.embed_short_text(keyword)
        return self._sort_by_vector_similarity(keyword_embedding, exclude_known_papers=[])


    def recommend_to_author(self, author_id):
        known_papers = self.profileDB.get_author_known_papers(author_id)
        author_embedding = self.profileDB.get_author_embedding(author_id, self.contentDB)
        exclude_known_papers = []
        for paper_set in known_papers.values():
            exclude_known_papers += list(paper_set)
        # recommend papers based on cosine similarity of user and paper
        recommend = self._sort_by_vector_similarity(author_embedding, exclude_known_papers)
        return recommend

    def recommend_by_weighted_linear_model(self, author_id, cos_weight, ref_weight, cite_weight, influence_weight, year_weight):
        known_papers = self.profileDB.get_author_known_papers(author_id)
        author_embedding = self.profileDB.get_author_embedding(author_id, self.contentDB)
        exclude_known_papers = []
        for paper_set in known_papers.values():
            exclude_known_papers += list(paper_set)
        # recommend papers based on cosine similarity of user and paper
        recommend = self.weighted_linear_combination_ranking(author_embedding, exclude_known_papers,
                                                             cos_weight, ref_weight, cite_weight, influence_weight, year_weight)
        return recommend