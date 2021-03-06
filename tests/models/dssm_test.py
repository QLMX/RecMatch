# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/9/9 7:58 下午
from match.models import DSSM
from tests.utils import check_model, get_xy_fd, get_ml1_data, predict_embed, get_top_k


top_k = 5
BASEPATH = "/Users/qlmx/Files/Project/RS/data/ml-1m/"
result_dir = BASEPATH + "result.txt"

def test_DSSM(get_recall=False):
    model_name = "DSSM"

    train_data, test_data, user_feature_columns, item_feature_columns, index_data_dict = get_ml1_data()
    model = DSSM(user_feature_columns, item_feature_columns, )

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
    test_DSSM()