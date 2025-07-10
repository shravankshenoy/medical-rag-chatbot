import re
import json
import streamlit as st

from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


## Initialize Elasticsearch client

client = Elasticsearch('http://localhost:9200')
print(client.info())

## Initialize embedding model for KNN Search
embedding_model_name = "multi-qa-distilbert-cos-v1"
embedding_model = SentenceTransformer(embedding_model_name)

## Index Properties

index_name = "medical-questions"

index_template = {
    "settings":{"number_of_shards":1},
    "mappings":{
        # Type of each field in the data : Question, Answer, etc
        "properties":{
            "Question": {"type": "text"},
            "Answer": {"type": "text"},
            "Question Type":{"type": "keyword"},
            "id": {"type": "keyword"},
            "question_answer_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

## Create index
if not client.indices.exists(index=index_name):
    client.indices.create(index=index_name, body=index_template)

all_indices = client.indices.get_alias(index='*')
for index_name in all_indices:
    print(index_name)

## Load documents into index
def index_documents():
    with open("data/Medical-QA-100.json","r") as f:
        document = json.load(f)

    for doc in tqdm(document):
        client.index(index=index_name, document=doc)


## Query clauses for querying Elasticsearch    
keyword_query = {
    "query": {
        "simple_query_string":{
            "query": None,
            "fields":["Question", "Answer"],
        }
    },
    "size": None
}


knn_query = {
    "knn":{
            "field": "question_answer_vector", # name of field in data containing vector
            "query_vector": None,
            "k": 20, # the number of nearest neighbors to return from each shard.
            "num_candidates": 1000, # The number of nearest neighbor candidates to consider per shard while doing knn search
    },
    "size" : None
}

## Perform keyword search
def keyword_search(query, size=3):
    keyword_query['query']['simple_query_string']['query'] = query
    keyword_query['size'] = size
    
    keyword_results = client.search(index="medical-questions",
        body=keyword_query)['hits']['hits'] 
    #print([res['_source']['Question'] for res in keyword_results])
    #print(keyword_results[0])
    return keyword_results


keyword_search("parasites")

## Perform knn search
def knn_search(query, size=3):

    query_vector = embedding_model.encode(query)
    knn_query['knn']['query_vector'] = query_vector
    knn_query['size'] = size

    knn_results = client.search(
            index="medical-questions",
            body=knn_query
        )['hits']['hits']
    
    print([res['_source']['Question'] for res in knn_results])

    return knn_results

knn_search("parasites")

## Streamlit application
# st.title("Medical FAQ Assistant")
# query = st.text_input("Enter your question")
# if st.button("Enter"):
#     if query:
#         answer = keyword_search(query)
#         st.markdown("### Answer:")
#         st.write(answer)
    
    



