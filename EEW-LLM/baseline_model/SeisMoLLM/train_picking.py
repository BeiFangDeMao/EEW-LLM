#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SeisMoLLM 单通道 P 波拾取；数据 H5 通过 --h5 指定。"""
from seismollm_model import train_task

if __name__ == "__main__":
    train_task("picking")
