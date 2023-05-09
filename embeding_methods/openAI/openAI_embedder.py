import openai

from embeding_methods.abstract_segment_embedding import *
from paper_reader import *

# pip install openai


embedding_model = 'text-embedding-ada-002'  # cheapest option
my_openai_private_key_path = '/Users/samuelyiin/recommender_system/embeding_methods/openAI/my_openai_api_key'
with open(my_openai_private_key_path) as file:
    my_openai_private_key = file.read()


class OpenAIEmbedder(AbstractSegmentEmbedder):
    def __init__(self, text_len_limit):
        super().__init__(text_len_limit)
        self.paper_reader = CERMINE_Reader()
        # todo CERMINE will request from some db, so this will change
        openai.api_key = my_openai_private_key

    def _parse_paper(self, paper_fp) -> Dict:
        """pick a parser, the parser will return attribute dictionary"""
        attribute_dict = self.paper_reader.parse(paper_fp)
        # to know the keys of the dict is the responsibility of the embedder
        # todo now there is only one attribute: "content", the whole paper
        attribute_dict = self.paper_reader.trim_special_characters(attribute_dict)
        return attribute_dict

    def _embed_text(self, text_string) -> np.ndarray:
        """assume the text is already under legal length"""
        embedding = np.array(openai.Embedding.create(input=[text_string],
                                                     model=embedding_model)['data'][0]['embedding'])
        return embedding

    def embed_paper(self, paper_fp):
        attribute_dic = self._parse_paper(paper_fp)
        # todo first edition, assume attribute dict only have "content"
        content = attribute_dic['content']
        return self._embed_long_text(content)


# text_len_limit = 5600  # 100 tokens ~= 75 words, max = 8191
# paper_fp = '/Users/samuelyiin/recommender_system/papers_pdf/test2.pdf'
# embedder = OpenAIEmbedder(text_len_limit)
# embedder.embed_paper(paper_fp)


# test openai
# openai.api_key = my_openai_private_key
# text_string = "this is a test string"
# embedding = np.array(openai.Embedding.create(input=[text_string],
#                                              model=embedding_model)['data'][0]['embedding'])
