import re
import json

from tqdm import tqdm
from elasticsearch import Elasticsearch

## Initialize Elasticsearch client
client = Elasticsearch('http://localhost:9200')

print(client.info())

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
    with open("Medical-QA-100.json","r") as f:
        document = json.load(f)

    for doc in tqdm(document):
        client.index(index=index_name, document=doc)


## Query clause for querying Elasticsearch    
keyword_query = {
    "query": {
        "simple_query_string":{
            "query": None,
            "fields":["Question", "Answer"],
        }
    }
}

## Execute search and get results
def keyword_search(query):
    keyword_query['query']['simple_query_string']['query'] = query
    print(keyword_query)
    keyword_results = client.search(index="medical-questions",
        body=keyword_query, size=5)['hits']['hits']
    
    print([res['_source']['Question'] for res in keyword_results])
    return(keyword_results)


keyword_search("parasites")


    



