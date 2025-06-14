import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Your existing extract_text_from_json function
def extract_text_from_json(data, filename=""):
    extracted = []

    filename_lower = filename.lower()

    if "constant_check" in filename_lower:
        for item in data:
            title = item.get("title", "")
            link = item.get("link", "")
            desc = item.get("description", "")
            extracted.append(f"{title}: {desc} Link: {link}")

    elif "course_intro" in filename_lower:
        text = data.get("text", "")
        link = data.get("link", {})
        link_text = link.get("text", "")
        link_href = link.get("href", "")
        extracted.append(f"Course Intro: {text} [{link_text}]({link_href})")

    elif "course_links" in filename_lower:
        for item in data:
            text = item.get("text", "")
            href = item.get("href", "")
            extracted.append(f"{text} Link: {href}")

    elif "details_section" in filename_lower:
        for section in data:
            summary = section.get("summary", "")
            extracted.append(f"Summary: {summary}")
            for para in section.get("paragraphs", []):
                extracted.append(f"Paragraph: {para}")
            for bullet in section.get("bullets", []):
                extracted.append(f"Bullet: {bullet}")
            for link in section.get("links", []):
                text = link.get("text", "")
                href = link.get("href", "")
                extracted.append(f"Link: {text} ({href})")

    elif "sidebar_section" in filename_lower:
        flattened_data = []
        for item in data:
            if isinstance(item, list):
                flattened_data.extend(item)
            else:
                flattened_data.append(item)

        for section in flattened_data:
            title = section.get("title", "")
            href = section.get("href", "")
            extracted.append(f"Section: {title} | Link: {href}")
            children = section.get("children", [])
            for child in children:
                child_title = child.get("title", "")
                child_href = child.get("href", "")
                extracted.append(f"  Subsection: {child_title} | Link: {child_href}")

    elif "notes" in filename_lower:
        for section in data:
            for item in section:
                text = item.get("text", "")
                link = item.get("link", "")
                if link:
                    extracted.append(f"{text} (Link: {link})")
                else:
                    extracted.append(text)

    elif "people_help" in filename_lower:
        for role, members in data.items():
            for member in members:
                name = member.get("name", "")
                profile = member.get("profile_link", "")
                emails = ", ".join(member.get("emails", []))
                discourse = ", ".join(member.get("discourse_links", []))
                extracted.append(f"{role}: {name} | Profile: {profile} | Emails: {emails} | Discourse: {discourse}")

    elif "tables" in filename_lower:
        for table in data:
            rows = table.get("rows", [])
            headers = rows[0] if rows else []
            for row in rows[1:]:
                row_data = ", ".join([f"{headers[i]}: {row[i]}" for i in range(len(row))])
                extracted.append(f"Exam Entry: {row_data}")

    elif "jan_2025_links" in filename_lower:
        for item in data:
            text = item.get("text", "")
            href = item.get("href", "")
            extracted.append(f"{text} Link: {href}")


    elif "updates" in filename_lower:
        for ann in data:
            extracted.append(ann.get("text", ""))
            nested = ann.get("nested", [])
            if nested:
                extracted.extend(nested)

    else:
        extracted.append(json.dumps(data))

    return " ".join(extracted)

# Initialize model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Folder containing static knowledge JSON files
STATIC_KNOWLEDGE_FOLDER = "knowledge"

# Output data
static_documents = []
static_embeddings = []

for filename in os.listdir(STATIC_KNOWLEDGE_FOLDER):
    if filename.endswith(".json"):
        filepath = os.path.join(STATIC_KNOWLEDGE_FOLDER, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
                text = extract_text_from_json(content, filename)
                embedding = model.encode(text)
                static_documents.append({
                    "filename": filename,
                    "text": text
                })
                static_embeddings.append(embedding)
                print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save embeddings
output_data = {
    "documents": static_documents,
    "embeddings": np.array(static_embeddings)
}

with open("static_embeddings.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("Static embeddings saved to static_embeddings.pkl")
