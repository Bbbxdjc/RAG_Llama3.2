import tkinter as tk
from tkinter import scrolledtext
import docx
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama

os.environ['TRANSFORMERS_OFFLINE'] = '1'

# ----- Document Processing and Section Splitting Functions -----
def split_doc_by_heading_with_title(doc):
    sections = []
    current_section = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Check if the paragraph style indicates a heading (thus a title)
        if para.style.name.lower().startswith("heading"):
            if current_section is not None:
                sections.append(current_section)
            current_section = {"title": text, "content": ""}
        else:
            if current_section is None:
                # If no title appears first, create a default section named "Intro"
                current_section = {"title": "Intro", "content": ""}
            # Append the paragraph text to the current section content
            if current_section["content"]:
                current_section["content"] += "\n" + text
            else:
                current_section["content"] = text
    if current_section is not None:
        sections.append(current_section)
    return sections

def merge_sections_by_title(sections, model, threshold=0.8):
    """
    Merge sections based on cosine similarity of titles.
    If the cosine similarity between titles is above the threshold,
    the contents are merged and concatenated into full_text.
    """
    if not sections:
        return []
    merged = []
    used = [False] * len(sections)
    titles = [sec["title"] for sec in sections]
    title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=False)
    for i in range(len(sections)):
        if used[i]:
            continue
        merged_title = sections[i]["title"]
        merged_content = sections[i]["content"]
        used[i] = True
        for j in range(i+1, len(sections)):
            if used[j]:
                continue
            sim = util.cos_sim(title_embeddings[i], title_embeddings[j]).item()
            if sim >= threshold:
                merged_content += "\n" + sections[j]["content"]
                used[j] = True
        full_text = merged_title + "\n" + merged_content
        merged.append({"title": merged_title, "content": merged_content, "full_text": full_text})
    return merged

def tokenize(text):
    return text.lower().split()

def construct_prompt(query, top_reranked_text):
    prompt = (
        f"Background Information (ordered by importance, from most important to least important):\n{top_reranked_text}\n\n"
        f"Question: {query}\n\n"
        "Using all the information above, please generate a clear, comprehensive, and precise answer.\nAnswer:"
    )
    return prompt

# ----- Pre-load Document, Compute Embeddings, and Build Index -----
print("Loading document and computing embeddings. Please wait...")

# Please ensure that the document path is correct
doc = docx.Document("Advising FAQ (12-19-24 Update).docx")
sections_with_title = split_doc_by_heading_with_title(doc)

# Load the SentenceTransformer model (for generating embeddings)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
merged_sections = merge_sections_by_title(sections_with_title, embed_model, threshold=0.6)
full_texts = [sec['full_text'] for sec in merged_sections]

# Build BM25 index for the merged section texts
tokenized_texts = [tokenize(text) for text in full_texts]
bm25 = BM25Okapi(tokenized_texts)

# Compute and normalize embeddings for all sections
sections_embeddings = embed_model.encode(full_texts, convert_to_tensor=True, show_progress_bar=True)
sections_embeddings_norm = util.normalize_embeddings(sections_embeddings)

# Compute embeddings for section titles
titles = [sec["title"] for sec in merged_sections]
title_embeddings = embed_model.encode(titles, convert_to_tensor=True, show_progress_bar=True)

# Load CrossEncoder model for re-ranking
cross_encoder = CrossEncoder('ms-marco-MiniLM-L12-v2')

print("Initialization complete.")

# ----- Define Query Processing Function -----
def process_query(query):
    # Generate and normalize the query embedding
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    query_embedding_norm = util.normalize_embeddings(query_embedding)
    
    # Compute cosine similarity between the query and each section
    sections_cosine_scores = util.cos_sim(query_embedding_norm, sections_embeddings_norm)[0].cpu().numpy()
    
    # Compute BM25 scores
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    if np.max(bm25_scores) > 0:
        bm25_scores_norm = bm25_scores / np.max(bm25_scores)
    else:
        bm25_scores_norm = bm25_scores
    sections_score = 0.2 * bm25_scores_norm + 0.8 * sections_cosine_scores

    # Compute cosine similarity between the query and section titles
    query_title_embedding = embed_model.encode([query], convert_to_tensor=True)
    title_cosine_scores = util.cos_sim(query_title_embedding, title_embeddings)[0].cpu().numpy()

    sections_weight = 0.5  # Weight for section content
    title_weight = 0.5     # Weight for title
    overall_scores = sections_weight * sections_score + title_weight * title_cosine_scores

    # Rank sections in descending order and select the top 10 as candidates
    candidate_indices = np.argsort(-overall_scores)
    candidate_texts = []
    for idx in candidate_indices[:10]:
        candidate_texts.append("Title: " + merged_sections[idx]["title"] + "\nContent: " + merged_sections[idx]["content"])
    
    # Re-rank candidates using CrossEncoder
    cross_input = [[query, text] for text in candidate_texts]
    cross_scores = cross_encoder.predict(cross_input)
    rerank_order = np.argsort(-cross_scores)
    
    k = 5  # Select top 5 pieces of information as background
    top_reranked_text = "\n".join([candidate_texts[i] for i in rerank_order[:k]])
    
    # Construct the prompt for the Llama model
    enhanced_prompt = construct_prompt(query, top_reranked_text)
    
    # Call the Ollama API to get the answer from Llama
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": enhanced_prompt}])
    answer = response['message']['content']
    return answer

# ----- Build the UI -----
def on_query_submit():
    query = query_entry.get()
    if not query.strip():
        return
    # Display a processing message
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Processing query, please wait...\n")
    output_text.update()
    
    # Process the query and get the answer
    answer = process_query(query)
    
    # Display the answer in the output box
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, answer)
    output_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Llama Query Demo")

# Query input area
query_label = tk.Label(root, text="Please enter your query:")
query_label.pack(pady=5)

query_entry = tk.Entry(root, width=100)
query_entry.pack(pady=5)

submit_button = tk.Button(root, text="Submit Query", command=on_query_submit)
submit_button.pack(pady=5)

# Output area with scroll bar
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20, state=tk.DISABLED)
output_text.pack(pady=10)

root.mainloop()
