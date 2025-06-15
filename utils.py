from datetime import datetime
import re
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    """
    Returns an embedding function using OllamaEmbeddings.
    """
    embeddings = OllamaEmbeddings(model="bge-m3")
    return embeddings

def rerank_by_recency(docs_with_scores, decay_rate=0.01):
    """
    Reranks documents based on similarity score and recency derived solely from
    the year extracted from the filename using regex.
    Args:
        docs_with_scores (list): List of tuples containing documents and their scores.
        decay_rate (float): Rate at which the score decays based on recency.
    """
    pattern = r"(?<!\d)(20[2-9]\d)(?!\d)"
    now = datetime.now().astimezone()
    reranked = []

    for doc, score in docs_with_scores:
        age_in_days = None
        recency_boost = 1.0

        filename = doc.metadata.get("source") or ""
        filename = filename.split("/")[-1]  # Get the filename from the path
        match = re.search(pattern, filename)
        print(match)
        if match:
            year_str = match.group(1)
            try:
                # Assume Jan 1 of the extracted year for recency calculation
                creation_date = datetime(int(year_str), 1, 1, tzinfo=now.tzinfo)
                age_in_days = (now - creation_date).days
            except Exception:
                age_in_days = None

        if age_in_days is not None:
            recency_boost = 1 / (1 + decay_rate * age_in_days)

        final_score = score * recency_boost

        # Debug prints
        print(f"Filename: {filename}")
        print(f"Original Score: {score}")
        print(f"Year extracted: {match.group(1) if match else 'None'}")
        print(f"Age in days: {age_in_days}")
        print(f"Recency boost: {recency_boost}")
        print(f"Final Score: {final_score}")
        print("-" * 40)

        reranked.append((doc, final_score))

    # Sort by boosted score descending
    return sorted(reranked, key=lambda x: x[1], reverse=True)


