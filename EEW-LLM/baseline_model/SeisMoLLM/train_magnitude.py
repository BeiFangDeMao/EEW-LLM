#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SeisMoLLM 震级估计。"""
from seismollm_model import train_task

if __name__ == "__main__":
    train_task("magnitude")
