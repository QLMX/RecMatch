# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/10/15 12:39 上午
from match.models import YoutubeDNN
from tests.utils import check_model, get_xy_fd, get_ml1_data, predict_embed, get_top_k
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


top_k = 5
BASEPATH = "/Users/qlmx/Files/Project/RS/data/ml-1m/"
result_dir = BASEPATH + "result.txt"

def test_YoutubeDNN(get_recall=False):
    model_name = "YoutubeDNN"

    train_data, test_data, user_feature_columns, item_feature_columns, index_data_dict = get_ml1_data()
    model = YoutubeDNN(user_feature_columns, item_feature_columns)

    model.compile('adam', "binary_crossentropy")
    model.fit(train_data[0], train_data[1], batch_size=100, epochs=10, validation_split=0.5)
    model = check_model(model, model_name)
    if get_recall:
        test_item_emd, test_user_emd = predict_embed(model, model_name, test_data, False)
        result = get_top_k(test_item_emd, test_user_emd, test_data, index_data_dict, top_k)
        with open(result_dir, 'w') as f:
            for line in result:
                f.write(line + '\n')

if __name__ == "__main__":
    test_YoutubeDNN()