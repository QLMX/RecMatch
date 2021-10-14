# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/13 12:02 上午
import os
import numpy as np
from tensorflow.python.keras.models import Model, load_model, save_model
import faiss

from match.layers import custom_objects
from match.feature_column import SparseFeat, VarLenSparseFeat
from tests.data_loader import read_data, encode_data, gen_data_set, gen_model_input



def check_model(model, model_name, check_model_io=False):
    """
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param check_model_io: test save/load model file or not
    :return:
    """
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
    return model


def predict_embed(model, model_name, data, check_model_io=True):
    """
    predict test data to get result
    Args:
        model:
        model_name:
        data: user/item input
        is_user: Whether to obtain user characteristics
        check_model_io: save/load model or not
    Returns: embed
    """
    user_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    save_user_model_name = model_name + '_user'
    user_embs = user_model.predict(data[0], batch_size=2 ** 12)

    iter_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    save_item_model_name = model_name + '_item'
    item_embs = iter_model.predict(data[0], batch_size=2 ** 12)

    if check_model_io:
        save_model(model, save_item_model_name + '.h5')
        save_model(model, save_user_model_name + '.h5')

        np.save(save_item_model_name + 'data.npy', item_embs)
        np.save(save_user_model_name + 'data.npy', user_embs)
    return item_embs, user_embs

def get_top_k(user_emd, items_emd, data, index_data_dict, k):
    print("the items shape is: ", items_emd.shape)
    d = items_emd.shape[1]
    nlist = 10          # number of cluster centers
    m = 8               # number of bytes per vector
    quantizer = faiss.IndexFlatL2(d)
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(items_emd)
    index.add(items_emd)
    index.nprobe = 10  # make comparable with experiment above

    D, I = index.search(user_emd, k)
    result = []
    for i in range(len(I)):
        if data[0]["user_id"][i] not in index_data_dict["user_id"]:
            continue
        res_str = str(index_data_dict["user_id"][data[0]["user_id"][i]]) + ":"
        for j in range(k):
            if data[0]["movie_id"][I[i][j]] not in index_data_dict["movie_id"]:
                continue
            res_str += str(index_data_dict["movie_id"][data[0]["movie_id"][I[i][j]]]) + ',' + str(D[i][j]) + ";"
        result.append(res_str)
    return result

    # sub_bacth = 1024
    # index = len(user_emd) / sub_bacth + 1
    # for i in range(index):
    #     start = i * sub_bacth
    #     end = (i + 1) * sub_bacth
    #     D, I = index.search(user_emd[start:end], k)  # sanity check
    # result = {}




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

def get_ml1_data(SEQ_LEN=50, embedding_dim=16):
    data = read_data()
    data, user_profile, item_profile, user_item_list, feature_max_idx, index_data_dict = encode_data(data)
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
    return train_data, test_data, user_feature_columns, item_feature_columns, index_data_dict