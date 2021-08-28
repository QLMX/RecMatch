# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/23 12:37 上午
import pandas as pd

BASEPATH = "/Users/qlmx/Files/Project/RS/data/weixin/data/"
user_action_dir = BASEPATH + "user_action.csv"
feed_info_dir = BASEPATH + "feed_info.csv"
feed_emd_dir = BASEPATH + "feed_embeddings.csv"

def read_data():
    user_action = pd.read_csv(user_action_dir)
    feed_info = pd.read_csv(feed_info_dir)
    feed_emd = pd.read_csv(feed_emd_dir)

    return user_action, feed_info, feed_emd


if __name__ == "__main__":
    user, feed, feed_emd = read_data()
    print(user.head())
    print(feed.head())
    print(feed_emd.head())