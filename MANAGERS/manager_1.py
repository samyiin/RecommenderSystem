"""
This manager will read semantic scholar's data, and do recommend.
"""
import json
import pandas as pd
import numpy as np
from recommender_system.content_based.cosine_simularity import *

# todo setup a real database for larger storage and faster retrieve, change recommend accordingly.
# todo change embedding to real embedding, change openAI's embedding method: based on all attribute?

recommender = RSCosSimilarity()
recommended_papers = recommender.recommend(['history', 'law'])
print(recommended_papers[['title', 'similarity']])
