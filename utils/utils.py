# -- coding: utf-8 --
# Copyright (c) 2024 Yanlong Yang, https://github.com/yyxr75/RaLiBEV
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from loguru import logger


def set_log_dir(log_path):
    if not os.path.exists(log_path):
        logger.info("created log path!")
        os.makedirs(log_path)
    filename = os.path.join(log_path, 'train_log.txt')
    logger.add(filename)