from ContentBasedRS.ContentAnalyzer import *
from ContentBasedRS.ProfileLearner import *
from ContentBasedRS.FilteringComponent import *
from ContentBasedRS.Utils import *


def order_by_cosine_similarity(author_id=4565995):
    """Task one: recommend by cosine similarity"""
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # select an author from profile database
    result, _ = profileDB.query_database(
        f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    known_papers = BLOB_to_info(result[0][profileDB.get_col_index(profileDB.COL_KNOWN_PAPERS)])
    author_embedding = profileDB.get_author_embedding(known_papers, contentDB)
    exclude_known_papers = []
    for paper_set in known_papers.values():
        exclude_known_papers += list(paper_set)

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.vector_similarity(author_embedding, exclude_known_papers)
    print("---------------------------------------")


def order_by_numerical_field():
    """Task two: recommend by citation/referenced number"""
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.order_by_number(contentDB.COL_YEAR)
    print("---------------------------------------")

    recommend = recommender.order_by_number(contentDB.COL_REF_COUNT)
    print("---------------------------------------")

    recommend = recommender.order_by_number(contentDB.COL_CITE_COUNT)
    print("---------------------------------------")

    recommend = recommender.order_by_number(contentDB.COL_INFLUENTIAL_CITE_COUNT)
    print("---------------------------------------")


def ensemble(author_id=4565995):
    """task three: take many factor into consideration -> weighted linear combination ranking
    """
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # select an author from profile database
    result, _ = profileDB.query_database(
        f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    known_papers = BLOB_to_info(result[0][profileDB.get_col_index(profileDB.COL_KNOWN_PAPERS)])
    author_embedding = profileDB.get_author_embedding(known_papers, contentDB)
    exclude_known_papers = []
    for paper_set in known_papers.values():
        exclude_known_papers += list(paper_set)

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.weighted_linear_combination_ranking(author_embedding, exclude_known_papers)

    print("---------------------------------------")


def recommend_after_feedback(author_id=4565995):
    """Task one: recommend by cosine similarity"""
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # reset the selected author
    profileDB.clear_liked_papers(author_id)
    profileDB.commit_change()

    # select an author from profile database
    result, _ = profileDB.query_database(
        f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    known_papers = BLOB_to_info(result[0][profileDB.get_col_index(profileDB.COL_KNOWN_PAPERS)])
    author_embedding = profileDB.get_author_embedding(known_papers, contentDB)
    exclude_known_papers = []
    for paper_set in known_papers.values():
        exclude_known_papers += list(paper_set)

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.vector_similarity(author_embedding, exclude_known_papers)
    print("---------------------------------------")

    # # assume the user really likes the first 5 papers
    liked_papers = set(recommend[:5][contentDB.COL_PAPER_ID].tolist())
    new_known_papers = {profileDB.PAPER_KIND_LIKED: liked_papers}
    author_name = result[0][profileDB.get_col_index(profileDB.COL_NAME)]
    profileDB.update_author(4565995, author_name, new_known_papers)
    profileDB.commit_change()
    print("---------------------------------------")

    # give the second recommender
    # select an author from profile database
    result, _ = profileDB.query_database(
        f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    known_papers = BLOB_to_info(result[0][profileDB.get_col_index(profileDB.COL_KNOWN_PAPERS)])
    author_embedding = profileDB.get_author_embedding(known_papers, contentDB)
    exclude_known_papers = []
    for paper_set in known_papers.values():
        exclude_known_papers += list(paper_set)

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.vector_similarity(author_embedding, exclude_known_papers)
    print("---------------------------------------")


# test_create_paper_database()

# test_create_author_database()

# order_by_cosine_similarity()

# order_by_numerical_field()

# ensemble()

recommend_after_feedback()
