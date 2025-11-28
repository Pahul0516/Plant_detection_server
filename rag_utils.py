import os
from typing import List, Dict, Any

import lancedb
from langchain_openai import OpenAIEmbeddings

DB_DIR = "data/lancedb"
TABLE_NAME = "plant_care_guides"


class PlantCareRAG:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.db = lancedb.connect(DB_DIR)
        if TABLE_NAME not in self.db.table_names():
            raise RuntimeError(f"LanceDB table '{TABLE_NAME}' not found. Build it first.")
        self.table = self.db.open_table(TABLE_NAME)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)

    def query_by_label(self, plant_label: str, k: int = 5) -> List[Dict[str, Any]]:
        vector = self.embeddings.embed_query(plant_label)

        results = (
            self.table.search(vector, query_type="vector")  # use vector search
            .where(f"plant_label = '{plant_label}'")
            .limit(k)
            .to_list()
        )

        # If label-specific yields too few, broaden without filter
        if len(results) < k:
            broadened = (
                self.table.search(plant_label, query_type="text", embedding=self.embeddings)
                .limit(k)
                .to_list()
            )
            # Merge while preserving order and uniqueness
            dedup = {}
            for r in results + broadened:
                dedup[(r.get("source"), r.get("text"))] = r
            results = list(dedup.values())[:k]
        return results

    def format_context(self, records: List[Dict[str, Any]]) -> str:
        blocks = []
        for r in records:
            src = r.get("source", "")
            txt = r.get("text", "")
            blocks.append(f"Source: {src}\n{txt}")
        return "\n\n---\n\n".join(blocks)
