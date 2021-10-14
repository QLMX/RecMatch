# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/23 12:37 上午
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


BASEPATH = "/Users/qlmx/Files/Project/RS/data/ml-1m/"
movie_dir = BASEPATH + "movies.dat"
rating_dir = BASEPATH + "ratings.dat"
user_dir = BASEPATH + "users.dat"

def read_data():
    '''
    load ori_data
    Returns: csv_data
    '''
    movie_columns = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(movie_dir, sep='::', header=None, names=movie_columns, engine='python')

    rating_columns = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_csv(rating_dir, sep='::', header=None, names=rating_columns, engine='python')

    user_columns = ['user_id','gender','age','occupation','zip']
    users = pd.read_csv(user_dir, sep='::', header=None, names=user_columns, engine='python')

    data = pd.merge(ratings, movies)
    data = pd.merge(data, users)

    return data

def encode_data(data):
    '''
    encode the feature used
    Args:
        data: data_csv
    Returns:
    '''
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_id = {}
    ori_user_ids = data['user_id'].copy().values
    ori_movie_ids = data['movie_id'].copy().values
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_id[feature] = data[feature].max() + 1

    index_data_dict = {}
    index_data_dict['user_id'] = {}
    index_data_dict['movie_id'] = {}
    pre_users_ids = data['user_id'].values
    pre_movie_ids = data['movie_id'].values
    for i in range(len(pre_users_ids)):
        if pre_users_ids[i] not in index_data_dict['user_id']:
            index_data_dict['user_id'][pre_users_ids[i]] = ori_user_ids[i]

        if pre_movie_ids[i] not in index_data_dict['movie_id']:
            index_data_dict['movie_id'][pre_movie_ids[i]] = ori_movie_ids[i]

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)
    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    return data, user_profile, item_profile, user_item_list, feature_max_id, index_data_dict


def gen_data_set(data, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1 , len(hist[::-1]), rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]), len(test_set[0]))

    return train_set, test_set

def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label



if __name__ == "__main__":
    data = read_data()
    data, user_profile, item_profile, user_item_list, feature_max_id = encode_data(data)
    train_set, test_set = gen_data_set(data, 10)
    train_model_input, train_label = gen_model_input(train_set, user_profile, 50)