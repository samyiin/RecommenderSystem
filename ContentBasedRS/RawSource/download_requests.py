import json
import os
import urllib

import requests
# get list of available release
r1 = requests.get('https://api.semanticscholar.org/datasets/v1/release').json()
print(r1[-100:])
S2_API_KEY = os.getenv("S2_API_KEY")


def download_release(release):
    papers = requests.get(f"http://api.semanticscholar.org/datasets/v1/release/{release}/dataset/papers",
                          headers={'x-api-key': S2_API_KEY}).json()
    urllib.request.urlretrieve(papers['files'][0], f"bulk_download/2023-06-06/papers-part0.jsonl.gz")
    print('success!')


def find_basis_paper():
    query = 'machine learning'
    S2_API_KEY = os.getenv('S2_API_KEY')    # export S2_API_KEY=xxxxx (no space for equal sign)
    offset = 9900
    result_limit = 99
    # for _ in range(3000):
    basic_info = 'paperId,title,authors,abstract,references.paperId,'  # basic information of paper
    rank_info = 'year,referenceCount,citationCount,influentialCitationCount'  # fields that can be used as ranking
    fields = basic_info + rank_info

    rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                       headers={'x-api-key': S2_API_KEY},
                       params={'query': query, 'offset': offset, 'limit': result_limit, 'fields': fields})
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    papers = results['data']
    with open(f"papers/{query}-{offset}-{offset + result_limit -1}.json", "w") as file:
        json.dump(results['data'], file)
    print(f'offset = {offset}: success!')

        # offset += result_limit


def find_specter_embeddings(paper_id):
    S2_API_KEY = os.getenv('S2_API_KEY')  # export S2_API_KEY=xxxxx (no space for equal sign)
    fields = "embedding"

    rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/' + paper_id,
                       headers={'x-api-key': S2_API_KEY},
                       params={ 'fields': fields})
    rsp.raise_for_status()
    results = rsp.json()
    embedding = results['embedding']['vector']
    print(embedding)

download_release(r1[-1])