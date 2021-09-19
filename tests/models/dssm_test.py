# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/9/9 7:58 下午
from match.models import DSSM
from tests.utils import check_model, get_xy_fd, get_ml1_data

def test_DSSM():
    model_name = "DSSM"

    train_data, test_data, user_feature_columns, item_feature_columns = get_ml1_data()
    model = DSSM(user_feature_columns, item_feature_columns)

    model.compile('adam', "binary_crossentropy")
    check_model(model, model_name, train_data[0], train_data[1])


if __name__ == "__main__":
    test_DSSM()