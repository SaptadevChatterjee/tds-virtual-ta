import os
import json
import pickle
from sentence_transformers import SentenceTransformer

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

knowledge_base = []
kb_dir = "knowledge_base"

for file in os.listdir(kb_dir):
    if file.endswith(".json"):
        with open(os.path.join(kb_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

            thread_url = data.get("url")

            # Process main_post
            main_post = data.get("main_post", {})
            if main_post:
                content = main_post.get("content", "")
                post_id = main_post.get("post_id", "")
                combined_text = f"[Main Post]\n{content}".strip()

                knowledge_base.append({
                    "post_id": post_id,
                    "text": combined_text,
                    "parent_link": thread_url
                })

            # Process replies
            replies = data.get("replies", [])
            for reply in replies:
                content = reply.get("content", "")
                post_id = reply.get("post_id", "")
                combined_text = f"[Reply]\n{content}".strip()

                knowledge_base.append({
                    "post_id": post_id,
                    "text": combined_text,
                    "parent_link": thread_url
                })

# Build embeddings
texts = [item["text"] for item in knowledge_base]
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings + metadata
with open("embeddings.pkl", "wb") as f:
    pickle.dump({
        "embeddings": embeddings,
        "documents": knowledge_base
    }, f)

print(f"Total post-level embeddings: {len(knowledge_base)}")
print("Embeddings built and saved successfully.")
