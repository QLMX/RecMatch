# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/13 12:02 上午
import os
import numpy as np
from tensorflow.python.keras.models import Model, load_model, save_model

from match.layers import custom_objects
from match.feature_column import SparseFeat, VarLenSparseFeat
from tests.data_loader import read_data, encode_data, gen_data_set, gen_model_input



def check_model(model, model_name, x, y, check_model_io=True):
    """
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y:
    :param check_model_io: test save/load model file or not
    :return:
    """

    model.fit(x, y, batch_size=10, epochs=2, validation_split=0.5)

    print(model_name + " test train valid pass!")

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(x, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  i in [0,k_max) if MIND
    print(model_name + " user_emb pass!")
    item_embs = item_embedding_model.predict(x, batch_size=2 ** 12)

    print(model_name + " item_emb pass!")

    model.save_weights(model_name + '_weights.h5')
    model.load_weights(model_name + '_weights.h5')
    os.remove(model_name + '_weights.h5')
    print(model_name + " test save load weight pass!")
    if check_model_io:
        save_model(model, model_name + '.h5')
        model = load_model(model_name + '.h5', custom_objects)
        os.remove(model_name + '.h5')
        print(model_name + " test save load model pass!")

    print(model_name + " test pass!")
    # print(1)
    #
    # save_model(item_embedding_model, model_name + '.user.h5')
    # print(2)
    #
    # item_embedding_model = load_model(model_name + '.user.h5', custom_objects)
    # print(3)
    #
    # item_embs = item_embedding_model.predict(x, batch_size=2 ** 12)
    # print(item_embs)
    # print("go")


def get_xy_fd(hash_flag=False):
    user_feature_columns = [SparseFeat('user', 3), SparseFeat(
        'gender', 2), VarLenSparseFeat(
        SparseFeat('hist_item', vocabulary_size=3 + 1, embedding_dim=4, embedding_name='item'), maxlen=4,
        length_name="hist_len")]
    item_feature_columns = [SparseFeat('item', 3 + 1, embedding_dim=4, )]

    uid = np.array([0, 1, 2, 1])
    ugender = np.array([0, 1, 0, 1])
    iid = np.array([1, 2, 3, 1])  # 0 is mask value

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    hist_len = np.array([3, 3, 2, 1])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid,
                    'hist_item': hist_iid, "hist_len": hist_len}

    # feature_names = get_feature_names(feature_columns)
    x = feature_dict
    y = np.array([1, 1, 1, 1])
    return x, y, user_feature_columns, item_feature_columns

def get_ml1_data(SEQ_LEN=50, embedding_dim=32):
    data = read_data()
    data, user_profile, item_profile, user_item_list, feature_max_idx = encode_data(data)
    train_set, test_set = gen_data_set(data, 10)
    train_data = gen_model_input(train_set, user_profile, 50)
    test_data = gen_model_input(test_set, user_profile, 50)

    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]
    return train_data, test_data, user_feature_columns, item_feature_columns