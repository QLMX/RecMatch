import tensorflow as tf

from match.layers.activation import Dice, activation_layer
from match.layers.core import PredictionLayer, Similarity, DNN, PoolingLayer, SampledSoftmaxLayer, EmbeddingIndex
from match.layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from match.layers.utils import (NoMask, Hash, concat_func, reduce_mean, reduce_sum, reduce_max,
                                div, softmax, combined_dnn_input)


custom_objects = {'tf': tf,
                  'Dice':Dice,
                  'activation_layer':activation_layer,
                  'PredictionLayer': PredictionLayer,
                  'Similarity': Similarity,
                  'DNN':DNN,
                  'PoolingLayer':PoolingLayer,
                  'SampledSoftmaxLayer':SampledSoftmaxLayer,
                  'EmbeddingIndex':EmbeddingIndex,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  'NoMask': NoMask,
                  'Hash': Hash,
                  'concat_func': concat_func,
                  'reduce_mean': reduce_mean,
                  'reduce_sum': reduce_sum,
                  'reduce_max': reduce_max,
                  'div': div,
                  'softmax': softmax,
                  'combined_dnn_input':combined_dnn_input
                   }