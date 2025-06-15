import argparse
import os
import shutil
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import get_embedding_function
from langchain_community.vectorstores import Chroma


DATA_PATH = r"C:\Users\masta\OneDrive\Desktop\Antonio Mastroianni\Jobs\Interview_Assignments\Osservatori Digital\02\project\data"
CHROMA_PATH = r"C:\Users\masta\OneDrive\Desktop\Antonio Mastroianni\Jobs\Interview_Assignments\Osservatori Digital\02\project\chroma_database"

def main():
# Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument(
       "--clear",
       action = "store_true",
       help = "Reset the database"
    )
    args = parser.parse_args()

    if args.clear:
       print("Clearing database")
       clear_database()

    # Create or update the database
    documents = load_documents()
    for doc in documents:
        print(f"First 200 documents: {doc.page_content[:200]}")  # Preview content

    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
  document_loader = PyPDFDirectoryLoader(DATA_PATH)
  return document_loader.load()

def split_documents(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 50,
      length_function = len,
      is_separator_regex = False
  )
  return text_splitter.split_documents(documents)



def add_to_chroma(chunks: list[Document]):
  # Load existing Chroma DataBase
  db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function()
  )

  # Calculate page IDs
  chunks_with_ids = calculate_chunk_ids(chunks)

  # Add or update documents in the Chroma database
  existing_items = db.get(include=[])
  existing_ids = set(existing_items['ids'])
  print(f"Number of existing documents in DB: {len(existing_ids)}")

  # Onlu add new documents that don't already exist in the database
  new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

  if len(new_chunks):
    print(f"Adding {len(new_chunks)} new documents to the database.")
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunk_ids)
    print(f"Database now contains {len(db.get(include=[])['ids'])} documents.")

  else:
    print("No new documents to add to the database.")


def calculate_chunk_ids(chunks):
    """Creates unique IDs for each chunk based on its metadata.
    Args:
        chunks (list[Document]): List of Document objects to process.
    Returns:
        list[Document]: List of Document objects with updated metadata containing unique IDs.
    Example:
        data/monopoly.pdf:6:2
        Page : Source : Chunk Index
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
       source = chunk.metadata.get("source")
       page = chunk.metadata.get("page")
       current_page_id = f"{source}:{page}"

       # If the page ID is the same as the last one, increment the index.
       if current_page_id == last_page_id:
          current_chunk_index += 1
       else:
          current_chunk_index = 0
        
       # Calculate chunk ID
       chunk_id = f"{current_page_id}:{current_chunk_index}"
       last_page_id = current_page_id

       # Add to page metadata
       chunk.metadata["id"] = chunk_id

    return chunks

def calculate_chunk_year(chunks):
    """Calculates the year for each chunk based on its metadata.
    Args:
        chunks (list[Document]): List of Document objects to process.
    Returns:
        list[Document]: List of Document objects with updated metadata containing the year.
    """
    for chunk in chunks:
        source = chunk.metadata.get("source")

        # Extract year from the source filename
        pattern = r"(?<!\d)(20[2-9]\d)(?!\d)"
        year = re.search(pattern, source.split('/')[-1]).group(1)
        chunk.metadata["year"] = year

    return chunks


def clear_database():
    """Clears the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared the database at {CHROMA_PATH}.")
    else:
        print("No existing database to clear.")

if __name__ == "__main__":
    main()