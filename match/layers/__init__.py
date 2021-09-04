import tensorflow as tf

from match.layers.core import PredictionLayer, Similarity
from match.layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from match.layers.utils import (NoMask, Hash, concat_func, reduce_mean, reduce_sum, reduce_max,
                                div, softmax)


custom_objects = {'tf': tf,
                  'PredictionLayer': PredictionLayer,
                  'Similarity': Similarity,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  'NoMask': NoMask,
                  'Hash': Hash,
                  'concat_func': concat_func,
                  'reduce_mean': reduce_mean,
                  'reduce_sum': reduce_sum,
                  'reduce_max': reduce_max,
                  'div': div,
                  'softmax': softmax
                   }