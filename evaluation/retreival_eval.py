from app import keyword_search, knn_search
from tqdm import tqdm

import pandas as pd

def get_reciprocal_rank(relevance_list):
    if True in relevance_list:
        rank = relevance_list.index(True)
        reciprocal_rank = 1.0 / (rank + 1)
        return reciprocal_rank
    
    return 0

def get_mrr(relevance_2d_list):
    score = 0.0
    for relevance_list in relevance_2d_list:
        reciprocal_rank = get_reciprocal_rank(relevance_list)
        score = score + reciprocal_rank
    
    score = score / len(relevance_2d_list)

    return score

def hit_rate():
    pass

def evaluate(ground_truth, search_engine, size=5):
    relevance_total = []
    i = 0
    for q in tqdm(ground_truth):
        doc_id = q['document']
        query = q['question']
        results = search_engine(query, size)
        print(results[1])
        relevance = [d['_source']['id'] == doc_id for d in results]
        relevance_total.append(relevance)
        print(relevance_total)
        
        i = i+1
        if i == 7:
            break

    return {
        'mrr': get_mrr(relevance_total),
    }

df_ground_truth = pd.read_csv('ground-truth-data-100.csv')
ground_truth = df_ground_truth.to_dict(orient='records')
keyword_results = evaluate(ground_truth, keyword_search)
knn_results = evaluate(ground_truth, knn_search)
headers = ["search method", "mean reciprocal rank"]

data = [
    ["keyword search", keyword_results["mrr"]],
    ["knn_search"], knn_results["mrr"]
]

print(data)