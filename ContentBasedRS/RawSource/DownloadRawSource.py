import json

import requests

features = 'paperId,corpusId,title,authors,embedding,publicationDate,referenceCount,citationCount,influentialCitationCount,tldr,abstract'
# download sample paper
r1 = requests.get(f'https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&offset=0&limit=90&fields={features}').json()

with open("papers/sample_papers.json", "w") as file:
    json.dump(r1['data'], file)



