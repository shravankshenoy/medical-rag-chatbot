import json
import hashlib

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

### This file loads dataset, generates embedding for each question-answer pair, and stores into json 

def generate_document_id(doc):
    combined = f"{doc['Question Type']}-{doc['Question']}-{doc['Answer'][:20]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:10]
    return document_id

model_name = "multi-qa-distilbert-cos-v1"
model = SentenceTransformer(model_name)
data = load_dataset("keivalya/MedQuad-MedicalQnADataset")
data = data["train"]
df = data.to_pandas()
df = df.rename(columns={"qtype": "Question Type"})
documents = df.to_dict(orient="records")

## Convert question answer pair to a vector embedding
for doc in tqdm(documents):
    qa_text = f"{doc['Question']} {doc['Answer']}"
    embed = model.encode(qa_text)
    doc["question_answer_vector"] = embed.tolist()
    doc["id"] = generate_document_id(doc)

with open("Medical-QA.json", "w") as f:
    json.dump(documents, f, indent=4)
    