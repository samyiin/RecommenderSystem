import json

import requests

# download sample paper
r1 = requests.get('https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&offset=0&limit=100&fields=title,authors,tldr,embedding,abstract').json()
print(r1)
with open("papers/sample_papers.json", "w") as file:
    json.dump(r1['data'], file)



