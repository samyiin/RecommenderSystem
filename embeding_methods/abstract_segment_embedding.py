from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class AbstractSegmentEmbedder(ABC):
    def __init__(self, text_len_limit):
        self.segment_length = text_len_limit


    def _embed_text(self, text_string) -> np.ndarray:
        """all subclass must implement this method"""
        pass

    def embed_paper(self, paper_attribute_dic):
        """all subclass must implement this method"""
        pass

    def _embed_long_text(self, long_text_string) -> np.ndarray:
        text_segments = self._sliding_window_segmentation(long_text_string)
        embedding_vectors = []
        for text in text_segments:
            embedding_vectors.append(self._embed_text(text))
        embedding = self._unweighted_avg_pooling(embedding_vectors)
        return embedding

    # solving the long text embed problem
    def _sliding_window_segmentation(self, long_text_string) -> List[str]:
        """The current approach is to use sliding windows + weighted average pooling"""
        # cut text into chunks
        segment_length = self.segment_length  # depend on which model to use
        window_size_proportion = 0.5
        text_segments = [long_text_string[i:i + segment_length]
                         for i in range(0, len(long_text_string), int(0.5 * segment_length))]
        return text_segments

    def _unweighted_avg_pooling(self, embedding_vectors: List[np.ndarray]) -> np.ndarray:
        """assume embeddings are the same dimension"""
        shape = embedding_vectors[0].shape
        weights = np.ones(shape) / shape[0]
        # Multiply each array with its corresponding weight
        weighted_arrays = np.multiply(weights, embedding_vectors)
        # Sum up the weighted arrays
        result = np.sum(weighted_arrays, axis=0)
        return result
