import argparse
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from utils import get_embedding_function, rerank_by_recency

CHROMA_PATH = "C:/Users/masta/OneDrive/Desktop/Antonio Mastroianni/Jobs/Interview_Assignments/Osservatori Digital/02/project/chroma_database"

PROMPT_TEMPLATE= """
Rispondi alla seguente domanda utilizzando **esclusivamente** le informazioni presenti nel contesto qui sotto. Se non trovi una risposta, dì che non è presente nel contesto:

Contesto:
{context}

---

Domanda: {question}

"""

def main():
    # Create CLI

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_text",
        type=str,
        help="The query text."
    )
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Load DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Search DB
    results = db.similarity_search_with_score(query_text, k=5)
    reranked_results = rerank_by_recency(results, decay_rate=0.01)
    top_docs = reranked_results[:3]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata["source"] for doc, _ in top_docs]
    formatted_response = f"Response: {response_text}\n\nSources: {sources} \n\n Scores: {[score for _, score in reranked_results[:3]]}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
