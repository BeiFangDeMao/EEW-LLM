#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SeisMoLLM 后方位角估计（sin/cos）。"""
from seismollm_model import train_task

if __name__ == "__main__":
    train_task("azimuth")
