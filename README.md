RAG with recency ranking system 

This project implements a document ranking system focused on boosting scores of recent documents using their year derived from the filenames. It processes PDF documents, extracts metadata including filename, and reranks search results based on document recency to improve relevance.

Features

- Parses PDF metadata to extract creation dates.
- Calculates document age in days relative to the current date.
- Applies a recency boost to raw similarity scores, prioritizing newer documents.
- Handles timezone-aware date parsing for accurate age calculation.
- Provides detailed debugging information for transparency.

Installation

Make sure you have Python 3.8+ installed. Then, install the required dependencies:

pip install -r requirements.txt

Usage

1. Place your PDF documents in the data folder.
2. Run the generate database script (e.g., python populate_database.py) to create the Chroma database.
3. Run the query_data.py with your prompt (e.g., python query_data.py "What is a digital wallet?".
4. Check console logs for detailed debugging info about scores, ages, and boosts.

Important Notes

- The system assumes documents have creationdate metadata in ISO8601 format with timezone info.
- Timezone-aware datetime handling avoids errors during age calculation.
- If year is missing or malformed, the recency boost defaults to 1 (no boost).
