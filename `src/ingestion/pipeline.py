```python
"""Ingestion pipeline: PDF/DOCX → chunks → Qdrant + BM25 index."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pymupdf4llm
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_pdf(path: Path) -> list[dict]:
    """Extract text with page numbers using pymupdf4llm."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [{"text": p["text"], "page": p["metadata"]["page"], "source": path.name}
            for p in pages if p["text"].strip()]


def chunk_documents(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for text in texts:
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            chunks.append({
                "id": chunk_id,
                "text": text,
                "source": page["source"],
                "page": page["page"]
            })
    return chunks


def ingest(docs_path: str) -> dict:
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create collection if not exists
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    all_chunks = []
    for pdf_path in Path(docs_path).glob("**/*.pdf"):
        pages = load_pdf(pdf_path)
        all_chunks.extend(chunk_documents(pages))

    # Save BM25 corpus
    corpus_path = Path("./data/bm25_corpus.json")
    corpus_path.parent.mkdir(exist_ok=True)
    with open(corpus_path, "w") as f:
        json.dump(all_chunks, f)

    # Upsert to Qdrant
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    from langchain.schema import Document
    docs = [Document(page_content=c["text"],
                     metadata={"source": c["source"], "page": c["page"], "id": c["id"]})
            for c in all_chunks]
    store.add_documents(docs)

    print(f"✓ Ingested {len(all_chunks)} chunks from {docs_path}")
    return {"chunks": len(all_chunks), "collection": COLLECTION}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()
    ingest(args.docs)
