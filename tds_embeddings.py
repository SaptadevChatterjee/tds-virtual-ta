import os
import json
import glob
from openai import OpenAI

client = OpenAI(api_key=os.getenv("AIPROXY_TOKEN"))

def extract_text(obj):
    texts = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()
            if isinstance(value, (dict, list)):
                texts.extend(extract_text(value))
            else:
                # Capture any potentially useful field
                if key_lower in ["summary", "summary_text", "summary_html", "title", "name", "text", "description"]:
                    texts.append(str(value).strip())
                elif key_lower in ["emails", "discourse_links", "profile_link", "link", "href"]:
                    texts.append(f"Link: {value}".strip())
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))
    elif isinstance(obj, str):
        texts.append(obj.strip())

    return texts

STATIC_PATH = "knowledge_base/tds_static_pages/*.json"
embedded_data = []

for filepath in glob.glob(STATIC_PATH):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"Error parsing {filepath}")
            continue

    all_texts = extract_text(data)
    full_text = " ".join(all_texts)
    full_text = " ".join(full_text.split())  # Clean redundant spaces

    # Split into chunks (important for large files)
    chunk_size = 1500
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    for chunk in chunks:
        if chunk.strip():
            response = client.embeddings.create(
                input=chunk.strip(),
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding

            embedded_data.append({
                "source": filepath,
                "content": chunk.strip(),
                "embedding": embedding
            })

# Save embeddings
with open("tds_static_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedded_data, f, ensure_ascii=False, indent=2)
