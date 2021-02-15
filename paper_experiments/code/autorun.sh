#!/usr/bin/env bash

python3 make_full_dag_eval_data.py
python3 run_dag_methods.py
python3 save_dag_metrics.py
