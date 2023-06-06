from ContentBasedRS.ContentAnalyzer import *
from ContentBasedRS.ProfileLearner import *
from ContentBasedRS.FilteringComponent import *
from ContentBasedRS.Utils import *



"""initialization: This should only run once!"""
def initialize_database():
    def _record_default_authors(row, args):
        """
        This is a callback for pandas dataframe.
        for each row: add the author to author table, if not exist, else, update the author
        todo: Gigantic room for speed improvement, not just realize the function
        """
        authors = BLOB_to_info(row[contentDB.COL_IDX_AUTHORS])
        paper_id = row[contentDB.COL_IDX_PAPER_ID]
        paper_embedding = BLOB_to_info(row[contentDB.COL_IDX_EMBEDDING])
        for author_info in authors:
            author_name = author_info[contentDB.ATTR_AUTHORS_NAME]
            author_id = author_info[contentDB.ATTR_AUTHORS_ID]
            # todo what todo when there is no author id?
            if author_id is None:
                continue
            profileDB.update_author(author_id, author_name, paper_id, profileDB.PAPER_KIND_WRITE, paper_embedding)

    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # CAUTION: Running this will delete all data!
    contentDB.create_main_table()
    profileDB.create_main_table()

    # record papers for content db in the main table
    raw_paper_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RawSource/papers'
    contentDB.add_papers_to_db(raw_paper_dir)

    # record default authors for profile db
    contentDB.for_each_row_do(_record_default_authors, args=[])

    # finish task, close connection
    profileDB.commit_change()
    contentDB.commit_change()
    profileDB.close_connection()
    contentDB.close_connection()


def order_by_cosine_similarity():
    """Task one: recommend by cosine similarity"""
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # select an author from profile database
    author_id = 2742129
    result, _ = profileDB.query_database(f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    author_embedding = BLOB_to_info(result[0][profileDB.COL_IDX_EMBEDDING])


    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.vector_similarity(author_embedding)

    # finish task, close connection
    profileDB.commit_change()
    contentDB.commit_change()
    profileDB.close_connection()
    contentDB.close_connection()

    print(recommend)

def order_by_field():
    """Task two: recommend by citation/referenced number"""
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.order_by_number(contentDB.COL_CITE_COUNT)

    # finish task, close connection
    profileDB.commit_change()
    contentDB.commit_change()
    profileDB.close_connection()
    contentDB.close_connection()

    print(recommend)


def ensemble():
    """task three: take many factor into consideration -> weighted linear combination ranking
    """
    # connect to contentDB and profile db
    profileDB = ProfileDB()
    contentDB = ContentDB()

    # select an author from profile database
    author_id = 2742129
    result, _ = profileDB.query_database(
        f"select * from {profileDB.MAIN_TABLE_NAME} where {profileDB.COL_AUTHOR_ID} = {author_id}")
    author_embedding = BLOB_to_info(result[0][profileDB.COL_IDX_EMBEDDING])

    # recommend papers based on cosine similarity of user and paper
    recommender = Recommender(contentDB, profileDB)
    recommend = recommender.weighted_linear_combination_ranking(author_embedding)

    # finish task, close connection
    profileDB.commit_change()
    contentDB.commit_change()
    profileDB.close_connection()
    contentDB.close_connection()

    print(recommend)

ensemble()