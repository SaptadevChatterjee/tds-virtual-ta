from fastapi import FastAPI, Request
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

client = OpenAI(
    api_key=AIPROXY_TOKEN,
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load discourse embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    discourse_embeddings = np.array(data["embeddings"])
    discourse_docs = data["documents"]

# Load static embeddings
with open("static_embeddings.pkl", "rb") as f:
    data_static = pickle.load(f)
    static_embeddings = np.array(data_static["embeddings"])
    static_docs = data_static["documents"]

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

    @validator('image', pre=True)
    def clean_base64(cls, v):
        if v is None:
            return v
        return re.sub(r'^[\'"]|[\'"]$', '', v).strip()

# Retrieval function
def retrieve_context(question: str, k_static: int = 10, k_discourse: int = 5):
    question_embedding = model.encode([question])[0]

    # Static similarity (FIRST priority)
    static_scores = np.dot(static_embeddings, question_embedding) / (
        np.linalg.norm(static_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    static_top_idx = np.argsort(static_scores)[-k_static:][::-1]
    static_matches = [static_docs[i]["text"] for i in static_top_idx]

    # Discourse similarity (SECOND priority)
    discourse_scores = np.dot(discourse_embeddings, question_embedding) / (
        np.linalg.norm(discourse_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    discourse_top_idx = np.argsort(discourse_scores)[-k_discourse:][::-1]
    top_discourse = []
    for idx in discourse_top_idx:
        doc = discourse_docs[idx]
        top_discourse.append({
            "text": doc["text"],
            "link": doc.get("parent_link"),
            "score": float(discourse_scores[idx])
        })

    return top_discourse, static_matches


@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        input = QuestionInput(**data)

        top_posts, static_matches = retrieve_context(input.question, k_static=10, k_discourse=5)

        context_text_discourse = "\n\n".join([p["text"] for p in top_posts])
        context_text_static = "\n\n".join(static_matches)
        final_context = f"{context_text_discourse}\n\n{context_text_static}"

        prompt_text = f"""
You are a helpful teaching assistant for IIT Madras’ Data Science program course called "Tools in Data Science".

Answer the question strictly using only the provided context.
Do not use any external knowledge. Do not make up any links.
Only use links that are present inside the provided context.

Context:
{final_context}

Question: {input.question}

Return your answer in the following JSON format:
{{
  "answer": "your full answer here",
  "links": [
    {{ "url": "...", "text": "..." }} ,
    ...
  ]
}}
"""

        messages = [{"role": "user", "content": prompt_text}]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        raw_output = response.choices[0].message.content

        try:
            parsed = json.loads(raw_output)
            answer_text = parsed.get("answer", raw_output)
            model_links = parsed.get("links", [])
        except:
            answer_text = raw_output
            model_links = []

        retrieved_links = []
        for post in top_posts:
            if post["link"]:
                retrieved_links.append({"url": post["link"], "text": post["text"]})

        final_links = {l["url"]: l for l in (model_links + retrieved_links) if "url" in l}.values()

        return {
            "answer": answer_text,
            "links": list(final_links)
        }

    except Exception as e:
        return {"error": str(e)}
