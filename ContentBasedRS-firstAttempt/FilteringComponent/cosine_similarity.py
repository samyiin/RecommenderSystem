import os.path

from ContentBasedRS_firstAttempt.ContentAnalyzer import *
from ContentBasedRS_firstAttempt.ProfileLearner import *
from ContentBasedRS_firstAttempt.tools import *
import numpy as np
import sys
project_root_dir = '/'.join(os.getcwd().split('/')[:-2])
sys.path.append(project_root_dir)


# pick openAI as the way to embed the papers
embedding_method = OpenAIEmbedding()
# test if openAI works
# todo it's ndarray(1536,) not (1, 1536) gotta change that!
# embedding = embedding_method.embed_long_text('this is the text to embed')

# actual recommender system
paper_source_dir = os.path.join(project_root_dir, 'ContentBasedRS-firstAttempt/TestInfoSource/papers')
represented_paper_dir = os.path.join(project_root_dir, 'ContentBasedRS-firstAttempt/StructuredData/RepresentedItems/')
citation_source_dir = os.path.join(project_root_dir, 'ContentBasedRS-firstAttempt/TestInfoSource/citations')
structured_citation_dir = os.path.join(project_root_dir, 'ContentBasedRS-firstAttempt/StructuredData/Citations')
user_source_dir = os.path.join(project_root_dir, 'ContentBasedRS-firstAttempt/TestInfoSource/authors/')
user_profiles_dir = os.path.join(project_root_dir,  'ContentBasedRS-firstAttempt/StructuredData/UserProfiles/')

# set up database for citations
citation_db = CitationDatabase()
citation_db.add_raw_info(citation_source_dir, structured_citation_dir)
# set up the database for papers to recommend
item_db = ItemDatabase(embedding_method, citation_db)
item_db.add_raw_info(paper_source_dir, represented_paper_dir)


# set up the database for users
profile_db = ProfileDatabase(embedding_method, item_db, user_profiles_dir)
profile_db.add_by_raw_info(user_source_dir)

# # pick a user
# user_id = "3165147"  # this user is from default user
# user_id = "102827218"  # this user is from default user, with paper 58033818
user_id = "2081021315"  # this user is from raw source
user = profile_db.query_database([profile_db.GET_BY_ID, user_id])  # user is in form of a dataframe (one row, x columns)
user_embedding = user.iloc[0][profile_db.EMBEDDING_COL]
user_liked_paper_list = user.iloc[0][profile_db.LIKED_PAPER_IDS_COL]  # todo should this be a set or a list?
print(f'pick user {user_id}')
print('this user likes these papers')
print(user_liked_paper_list)
print(f"this user's current embedding is {user_embedding}")


# query database of papers to find the papers with highest cosine similarity
result_limit_size = 10
result = item_db.query_database([item_db.COSINE_SIMILARITY, user_embedding, result_limit_size])
print(f'so we will feed this {result_limit_size} papers to this user')
print(result[[item_db.PAPER_ID_COL, item_db.PAPER_TITLE_COL]])

# collect user feedback and update the user profile in the user database
# suppose the user liked the first feed out of the first 10 posts
liked_paper = result.iloc[0][item_db.PAPER_ID_COL]
print(f'suppose this user liked paper id {liked_paper}')
user_liked_paper_list = np.append(user_liked_paper_list, liked_paper)
# update user's liked papers and embedding
user.at[0, profile_db.LIKED_PAPER_IDS_COL] = user_liked_paper_list
# user is a pandas df, user.iloc[0] is a pandas Series for the api of embed user
user.at[0, profile_db.EMBEDDING_COL] = profile_db.embed_user(user.iloc[0])
profile_db.update_user(user)
print("updating user profile......")

# test see if user profile is updated
user = profile_db.query_database([profile_db.GET_BY_ID, user_id])  # user is in form of a dataframe (one row, x columns)
print('this user likes these papers')
print(user_liked_paper_list)
print('user profile is updated to be the average of embeddings of these papers')
print(f"this user's current embedding is {user_embedding}")

print('----------------------------------------------------------------------------------------------------------')
