from ContentBasedRS.ContentAnalyzer import ItemDatabase
from ContentBasedRS.ProfileLearner import ProfileDatabase
from ContentBasedRS.tools import OpenAIEmbedding

embedding_method = OpenAIEmbedding()

paper_source_dir = '/cs/labs/avivz/avivz/semantic_scholar_data/papers/'
represented_items_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/RepresentedItems/'
item_db = ItemDatabase(paper_source_dir, represented_items_dir, embedding_method)

# info_source_dir = '/cs/labs/avivz/avivz/semantic_scholar_data/authors/'
# user_profiles_dir = '/cs/labs/avivz/hsiny/recommender_system/ContentBasedRS/UserProfiles/'
# profile_db = ProfileDatabase(info_source_dir, user_profiles_dir, item_db, embedding_method)
#
# user_id = 2081021315  # assuming such user exists todo fool proof
# user = profile_db.get_profile_by_id(user_id)   # user is in form of a dataframe (one row, x columns)
#
# paper_id = 58033818 # assuming such user exists todo fool proof
# paper = item_db.get_item_by_id(paper_id)
print(1)


