import numpy as np
from transformers import AutoTokenizer, AutoModel

# pip install openai
text_len_limit = 512  # 8191 tokens roughly equals to 5600 words
segment_length = text_len_limit


# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')


# concatenate title and abstract
def embed_short_text(text_string) -> np.ndarray:
    """embed text that's under legal length, """
    # preprocess the input
    inputs = tokenizer(text_string, padding=True, truncation=True, return_tensors="pt", max_length=text_len_limit)
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]
    embeddings = embeddings.squeeze().detach().numpy()
    return embeddings



def embed_long_text(long_text_string) -> np.ndarray:
    """embed text that's above legal length"""
    text_segments = _sliding_window_segmentation(long_text_string)
    embedding_vectors = []
    for text in text_segments:
        embedding_vectors.append(embed_short_text(text))
    embedding = _unweighted_avg_pooling(embedding_vectors)
    return embedding


# solving the long text embed problem
def _sliding_window_segmentation(long_text_string):
    """The current approach is to use sliding windows + weighted average pooling"""
    # cut text into chunks
    window_size_proportion = 0.5
    text_segments = [long_text_string[i:i + segment_length]
                     for i in range(0, len(long_text_string), int(0.5 * segment_length))]
    return text_segments


def _unweighted_avg_pooling(embedding_vectors) -> np.ndarray:
    """assume embeddings are the same dimension"""
    shape = embedding_vectors[0].shape
    weights = np.ones(shape) / shape[0]
    # Multiply each array with its corresponding weight
    weighted_arrays = np.multiply(weights, embedding_vectors)
    # Sum up the weighted arrays
    result = np.sum(weighted_arrays, axis=0)
    return result
