"""
Modern RAG Engine using:
- Gemini (google-genai SDK via wrapper)
- HuggingFace Embeddings
- Qdrant Vector DB
"""

import pandas as pd
from typing import List, Dict, Any

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .llm_gemini import GeminiLLM


class RAGEngine:
    def __init__(
        self,
        data_path: str,
        qdrant_url: str,
        collection_name: str,
        google_api_key: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.data_path = data_path
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.google_api_key = google_api_key

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = None
        self.vector_store = None
        self.storage_context = None
        self.index = None
        self.query_engine = None
        self.documents: List[Document] = []

    async def initialize(self):
        print("🚀 Starting Modern RAG Engine...")

        # -------------------------
        # QDRANT
        # -------------------------
        self.client = QdrantClient(url=self.qdrant_url)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # -------------------------
        # LLM (FIXED)
        # -------------------------
        Settings.llm = GeminiLLM(
            api_key=self.google_api_key,
            model="gemini-2.5-flash"
        )

        # -------------------------
        # EMBEDDINGS
        # -------------------------
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # -------------------------
        # CHUNKING
        # -------------------------
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # -------------------------
        # LOAD DATA
        # -------------------------
        await self.load_csv(self.data_path)

        print("✅ Modern RAG Engine ready!")

    async def load_csv(self, file_path: str):
        print(f"📄 Loading CSV: {file_path}")

        df = pd.read_csv(file_path)

        self.documents = [
            Document(
                text=" ".join(str(x) for x in row.values if pd.notna(x))
            )
            for _, row in df.iterrows()
        ]

        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
            show_progress=True
        )

        # ✅ improved retrieval
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3
        )

        print(f"📊 Loaded {len(self.documents)} documents into Qdrant")

    def query(self, question: str) -> str:
        if not self.query_engine:
            return "RAG not initialized"

        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            return f"Query failed: {str(e)}"

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "documents": len(self.documents),
            "collection": self.collection_name
        }