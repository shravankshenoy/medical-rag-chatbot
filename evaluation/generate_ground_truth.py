import json
import os
import pandas as pd

from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

with open("data/Medical-QA-100.json", "r") as f:
    documents = json.load(f)

prompt_template = """
You are tasked with generating potential questions based on the structure of a medical dataset. This dataset contains fields like Question Type, Question, and Answer. Formulate 5 possible questions that a user might ask based on the provided record. Each question should be complete, concise, and avoid directly using too many words from the record itself.

The record format:

Question Type: {Question Type}
Question: {Question}
Answer: {Answer}

Please provide the output in parsable JSON format without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()


client = Groq(api_key=GROQ_API_KEY)

def generate_questions(doc):
    prompt = prompt_template.format(**doc)
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    json_response = response.choices[0].message.content
    return json_response

results = {}
for doc in tqdm(documents):
    doc_id = doc['id']
    if doc_id in results:
        continue
    questions = generate_questions(doc)
    results[doc_id] = questions
    if len(results) >= 2000:
        break

parsed_results = {}
for doc_id, json_questions in results.items():
    try:
        parsed_results[doc_id] = json.loads(json_questions)
    except json.JSONDecodeError as e:
        continue

final_results = []

for doc_id, questions in parsed_results.items():
    for q in questions:
        final_results.append((q, doc_id))

df = pd.DataFrame(final_results, columns=['question', 'document'])
df.to_csv('ground-truth-data-100.csv', index=False)