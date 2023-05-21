from ContentBasedRS.ContentAnalyzer import *
from ContentBasedRS.ProfileLearner import *
from ContentBasedRS.tools import *
import numpy as np

embedding_method = OpenAIEmbedding()

paper_source_dir = '/cs/labs/avivz/avivz/semantic_scholar_data/papers/'
represented_items_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RepresentedItems/'
item_db = ItemDatabase(embedding_method)
item_db.add_to_database(paper_source_dir, represented_items_dir)

info_source_dir = '/cs/labs/avivz/avivz/semantic_scholar_data/authors/'
user_profiles_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/UserProfiles/'
profile_db = ProfileDatabase(item_db, embedding_method)
profile_db.add_to_database(info_source_dir, user_profiles_dir)

user_id = 2081021315  # assuming such user exists todo fool proof
user = profile_db.get_profile_by_id(user_id)  # user is in form of a dataframe (one row, x columns)
user_embedding = user.iloc[0][profile_db.EMBEDDING_COL]
# paper_id = 58033818 # assuming such user exists todo fool proof
# paper = item_db.get_item_by_id(paper_id)

result_limit_size = 10

result = item_db.query_database([item_db.COSINE_SIMILARITY, user_embedding, result_limit_size])
print(result)

# suppose the user reads the first feed
read_paper = result.iloc[0][item_db.PAPER_ID_COL]
user_liked_paper_list = user.iloc[0][profile_db.LIKED_PAPER_IDS_COL]  # todo should this be a set or a list?
user_liked_paper_list = np.append(user_liked_paper_list, read_paper)
user.at[0, profile_db.LIKED_PAPER_IDS_COL] = user_liked_paper_list
# user is a pandas df, user.iloc[0] is a pandas Series
user.at[0, profile_db.EMBEDDING_COL] = profile_db.embed_user(user.iloc[0])
profile_db.update_user(user)

user = profile_db.get_profile_by_id(user_id)  # get the user again and check if the status have been updated
print(1)


