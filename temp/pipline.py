from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models import Word2Vec
from sklearn.neighbors import DistanceMetric
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
import pickle
import numpy as np
import csv
from flask import Flask, request, jsonify, render_template

'''
first join up list of all 4514 rows of titles into a string
and tokennized this string, we get a list of 28 string, about same length
'''
colnames = ['no', 'id', 'title', 'abstract', 'citation', 'references']
data = pd.read_csv('papers.csv', names=colnames)
new_df = data[['id', 'title']]
col_titlesentences = data.title.tolist()  # join up all the titles
col_id = data.id.tolist()
# coverting list to string
str1 = ""
str1 = str1.join(col_titlesentences)
# feature engineering-remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize(str1)
sentences = nltk.sent_tokenize(str1)

'''
From above, we get a list of 4 strings, most of the words are in the first string
first separate sentence to words, stem the words, and then join the stemmed forms -> "stemmed sentences"
and then separate above sentence, lemmatize the words, and joined the lemmatized stemmed words -> "lemmatized sentences"
Note:
# stemmer: convert every word to its basic form
# lemmatizer: convert every word to its basic form, given POS (if not give, default noun)
'''
# text processing:
stemmer = PorterStemmer()
for i in range(len(sentences)):
    wordsStemmer = nltk.word_tokenize(sentences[i])
    wordsStemmer = [stemmer.stem(word) for word in wordsStemmer]
    sentences[i] = ' '.join(wordsStemmer)

# text processing two words are same then it will normalization
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    wordslemmatizer = nltk.word_tokenize(sentences[i])
    wordslemmatizer = [lemmatizer.lemmatize(word) for word in wordslemmatizer]
    sentences[i] = ' '.join(wordslemmatizer)

'''
now we have a list of 28 strings still, but all words in the string are stemmed and lemmatized
However, this dude take only the first string, 
and split it by period to form a list of 5 strings
And he somehow also decided to delete the last string of the 5 strings
'''
sentences = sentences[0].split('.')  # above research_paper_pdf.title.tolist() first word is column name "title"
del sentences[-1]  # why?

'''
(This part is unrelated to the text stemming/lematizing part above, it's taking directly from first section)
a matrix, columns are all 7520 possible kinds of words, rows are all 4514 titles, cell number is how many 
times the word appear in the sentence 
# dicti all 7520 possible words
# stop words: words that do not add meaning to sentence, like "the"
'''
stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stopWords)
featurevectors = vectorizer.fit_transform(col_titlesentences).todense()
dicti = vectorizer.vocabulary_


def word2vec(word2vecInput):
    # deleted words, unused......
    sentences = [[row] for row in col_titlesentences]
    model = Word2Vec(sentences, min_count=1, workers=3, window=3, sg=1, vector_size=50)  # delete attribute "size=50
    word2vecOutput = model.wv.most_similar(word2vecInput)
    return word2vecOutput


def build_model_knn(test2):
    neigh = NearestNeighbors(n_neighbors=5)
    global featurevectors
    featurevectors = np.asarray(featurevectors)  # featurevectors is a np.matrix
    neigh.fit(featurevectors)
    NearestNeighbors(algorithm='auto', leaf_size=30)

    final_knn = neigh.kneighbors(test2, return_distance=False)[0]
    IDS = []
    my_dict = {}
    for i in final_knn:
        my_dict[col_id[i]] = col_titlesentences[i]
        IDS.append(col_id[i])
    result = []
    for i in final_knn:
        result.append(col_titlesentences[i])

    return result


def recommend_collaborative_filtering(list_of_papers):
    target_references = []
    count = 0
    with open('papers.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        for row in readCSV:
            if count > 0:
                if row[2] in list_of_papers:
                    ref = row[6].split(',')
                    for r in ref:
                        target_references.append(r)
                    break
            count += 1

    id_title_dict = {}

    count = 0
    candidate_papers = []
    with open('papers.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if count > 0:

                id_title_dict[row[2]] = row[3]

                refs = row[6].split(',')
                for r in refs:
                    if r in list_of_papers:
                        candidate_papers.append(row[2])
                    if r in target_references:
                        candidate_papers.append(row[2])
                        break
            count += 1

    for paper in candidate_papers:
        if paper in list_of_papers:
            candidate_papers.remove(paper)

    candidate_papers_titles = []
    count = 0
    for id_ in candidate_papers:
        if count > 4:
            break
        count += 1
        candidate_papers_titles.append(id_title_dict.get(id_))

    return candidate_papers_titles


# cosine similarity
def cosine(test2):
    global featurevectors
    cosine_similarities = linear_kernel(test2, featurevectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-5:-1]
    # related_docs_indices_list = related_docs_indices.tolist()
    IDS = []
    my_dict = {}
    for i in related_docs_indices:
        my_dict[col_id[i]] = col_titlesentences[i]
        IDS.append(col_id[i])
    result = []
    for i in related_docs_indices:
        result.append(col_titlesentences[i])

    return result, IDS


# getting the research_paper_pdf
data_frame = pd.read_csv('papers.csv', index_col=False)
data_frame = data_frame.loc[:, ~data_frame.columns.str.match('Unnamed')]

# Getting only the required research_paper_pdf like id and title
new_data_frame = data_frame[['id', 'title']]

# Making into vectors
tfidfvectorizer = TfidfVectorizer()
tfidfmatrix = tfidfvectorizer.fit_transform(new_data_frame['title'])

data_frame = pd.DataFrame(tfidfmatrix.toarray())

# Caluculating similarity
cosine_sim = cosine_similarity(data_frame)
df_cosineSim = pd.DataFrame(cosine_sim)  # 4513 x 4513(number of sentence): similarity pair wise


# Recommendations
def recommendations(title, cosine_sim=cosine_sim):
    """
    what algorithm does it use?
    如果搜索的词刚好是名字，那就返回名字+10个与这一篇最相近的
    """
    recommended_titles = []
    recommended_id = []
    # what if title we search is not in new_data_frame? should return
    idx = new_data_frame[new_data_frame['title'].str.contains(title, case=False)]
    if len(idx) == 0:
        return None, None
    else:
        idx = idx.index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    top_10_indexes = list(score_series.iloc[1:6].index)
    uniq = []
    for i in top_10_indexes:
        recommended_titles.append(list(new_data_frame['title'])[i])
        recommended_id.append(list(new_data_frame['id'])[i])

        # recommended_titles[new_df['id'][i]] = new_df['title'][i]
        [uniq.append(x) for x in recommended_titles if x not in uniq]

    return (uniq, recommended_id)


def pipeline(title_name):
    """title name is a list of strings, for whih you want to search for the type of papers to recommend"""
    test2 = vectorizer.transform(title_name).toarray()
    '''since database is small, most of the words are not in it'''
    if not np.any(test2):
        print('we dont have this word')
        return
    """test 2 is a numpy ndarray"""
    knn_result = build_model_knn(test2)
    print('knn result')
    print(knn_result[1:])
    cosine_result, ids = cosine(test2)
    print('cosine similarity result')
    print(cosine_result)
    word2vec_result = word2vec(cosine_result[0])
    print('word to vec result')
    print(word2vec_result)
    uniq, recommended_id = recommendations(title_name[0])
    print('existing papers result')
    print(uniq)
    print('collaborative, item based result')
    result_collaborative = recommend_collaborative_filtering(ids)
    print(result_collaborative)
    # return knn_result, uniq, word2vec_result, result_collaborative


# int_features = [x for x in request.form.values()]
'''since database is small, most of the words are not in it'''
# select a word from dicti (existing words)
import random
idx = random.randint(0, len(dicti.keys()))
word = list(dicti.keys())[idx]
print('the word I choose is ' + word)
int_features = [word]
pipeline(int_features)

