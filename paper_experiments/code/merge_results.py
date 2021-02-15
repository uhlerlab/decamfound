import pickle

a = pickle.load(open('../results/synthetic/eval_removal_parent_set_results_lrps.pkl', 'rb'))
b = pickle.load(open('../results/synthetic/eval_removal_parent_set_results_temp.pkl', 'rb'))
print(list(a.values())[0].keys())
print(list(b.values())[0].keys())
print(len(a))
print(len(b))
print(set(a.keys()) >= set(b.keys()))
merged_results_all_keys = {k: {**b[k], **a[k]} for k in b.keys()}
merged_results_shared_keys = {k: {**b[k], **a[k]} if k in b else a[k] for k in a.keys()}
pickle.dump(merged_results_all_keys, open('../results/synthetic/eval_removal_parent_set_results_all_keys.pkl', 'wb'))
pickle.dump(merged_results_shared_keys, open('../results/synthetic/eval_removal_parent_set_results_shared_keys.pkl', 'wb'))
print(len(merged_results_all_keys))
print(len(merged_results_shared_keys))
print(list(merged_results_all_keys.values())[0].keys())
print(list(merged_results_shared_keys.values())[0].keys())
