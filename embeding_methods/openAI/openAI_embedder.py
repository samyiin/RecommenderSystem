import openai

from embeding_methods.abstract_segment_embedding import *
from paper_parser import *

# pip install openai
embedding_model = 'text-embedding-ada-002'  # cheapest option, 8k token at a time
my_openai_private_key = input("input your openAI api-key: \n")


class OpenAIEmbedder(AbstractSegmentEmbedder):
    def __init__(self, text_len_limit):
        super().__init__(text_len_limit)
        openai.api_key = my_openai_private_key

    def _embed_text(self, text_string) -> np.ndarray:
        """assume the text is already under legal length"""
        embedding = np.array(openai.Embedding.create(input=[text_string],
                                                     model=embedding_model)['data'][0]['embedding'])

        return embedding

    def embed_paper(self, paper_attribute_dic):
        """
        the paper will be represented in an attribute dictionary of semantic scholar's format
        By attribute dict, it could mean anything that can be access by __getitem__, such as a pandas row with correct
        column name, or an actual dictionary. Must include columns:
        Title, Author, content, citations
        """
        content = paper_attribute_dic['content']    # should just concatnate
        return self.embed_long_text(content)


# text_len_limit = 5600  # 100 tokens ~= 75 words, max = 8191
# paper_fp = '/Users/samuelyiin/recommender_system/papers_pdf/test2.pdf'
# embedder = OpenAIEmbedder(text_len_limit)
# embedder.embed_paper(paper_fp)


# test openai
# openai.api_key = my_openai_private_key
# text_string = "test string"
# embedding = np.array(openai.Embedding.create(input=[text_string],
#                                              model=embedding_model)['data'][0]['embedding'])
