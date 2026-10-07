#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SeisMoLLM 震中距估计（km）。"""
from seismollm_model import train_task

if __name__ == "__main__":
    train_task("distance")
