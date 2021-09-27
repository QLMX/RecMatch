# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/13 12:02 上午

from match.models import FM
from tests.utils import check_model, get_xy_fd, get_ml1_data


def train_FM(get_top_k=False):
    model_name = "FM"

    train_data, test_data, user_feature_columns, item_feature_columns = get_ml1_data()
    model = FM(user_feature_columns, item_feature_columns, )

    model.compile('adam', "binary_crossentropy")
    model = check_model(model, model_name, train_data[0], train_data[1])
    if get_top_k:
        test_item_emd, test_user_emd = predict_embed(model, model_name, test_data[0], test_data[1], False)



if __name__ == "__main__":
    train_FM(True)