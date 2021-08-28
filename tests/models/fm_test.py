# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : QLMX
# @Email   : wenruichn@gmail.com
# @Web     : www.growai.cn
# @Time    : 2021/8/13 12:02 上午

from deeprecall.models import FM
from tests.utils import check_model, get_xy_fd


def test_FM():
    model_name = "FM"

    x, y, user_feature_columns, item_feature_columns = get_xy_fd(False)
    model = FM(user_feature_columns, item_feature_columns, )

    model.compile('adam', "binary_crossentropy")
    check_model(model, model_name, x, y,)


if __name__ == "__main__":
    test_FM()
    # pass
