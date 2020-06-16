#!/bin/bash

model="model_80_DSS"
result_folder="DSS"

PYTHONIOENCODING=utf-8 python -u test.py checkpoints/$model >> result/$result/$model.out