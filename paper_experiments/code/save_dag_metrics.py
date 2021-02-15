import pickle
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from utils.metrics import dag_sample_metrics

results = pickle.load(open('../results/eval_dag_results.pkl', 'rb'))
for setting, r in results.items():
    print(f"== {setting} ==")
    candidate_dags = r['candidate_dags']
    vanilla_scores = r['vanilla_scores'][0]
    poet_scores = r['poet_scores'][0]

    print("Vanilla")
    dag_sample_metrics(candidate_dags, vanilla_scores)
    print("POET")
    dag_sample_metrics(candidate_dags, poet_scores)


# TODO: remove more edges, get worse results for POET
# TODO: move to server and run
