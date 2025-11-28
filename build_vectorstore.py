import os
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document

import lancedb
import pyarrow as pa
from langchain_openai import OpenAIEmbeddings


PLANT_URLS: Dict[str, List[str]] = {
    "orange": [
        "https://en.wikipedia.org/wiki/Orange_(fruit)"
    ],
    "banana": [
        "https://en.wikipedia.org/wiki/Banana"
    ]
}

DB_DIR = "data/lancedb"
TABLE_NAME = "plant_care_guides"

# Dimension of the embedding model you are using
VECTOR_DIM = 1536  # for text-embedding-3-small/ada-002


def load_and_split(urls_by_label: Dict[str, List[str]]) -> List[Document]:
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
    )

    for label, urls in urls_by_label.items():
        for url in urls:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata = {
                    **doc.metadata,
                    "source": url,
                    "plant_label": label
                }
            chunks = splitter.split_documents(loaded)
            docs.extend(chunks)

    return docs


def build_lancedb(docs: List[Document], api_key: str):
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    # Explicit schema with fixed-size vector column
    schema = pa.schema([
        ("text", pa.string()),
        ("source", pa.string()),
        ("plant_label", pa.string()),
        ("vector", pa.list_(pa.float32(), VECTOR_DIM))
    ])

    table = db.create_table(TABLE_NAME, schema=schema)

    embeddings = OpenAIEmbeddings(api_key=api_key)

    BATCH = 256
    batch = []
    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        batch.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "plant_label": doc.metadata.get("plant_label", ""),
            "vector": vector,
        })
        if len(batch) >= BATCH:
            table.add(batch)
            batch = []
    if batch:
        table.add(batch)

    print(f"Built LanceDB table at {DB_DIR}/{TABLE_NAME}")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for embeddings.")
    docs = load_and_split(PLANT_URLS)
    build_lancedb(docs, api_key)
