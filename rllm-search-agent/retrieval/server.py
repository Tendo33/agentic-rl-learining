#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

Usage:
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000
"""

import argparse
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


class LocalRetriever:
    """Dense-only retrieval system using FAISS."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.corpus = []
        self.dense_index = None
        self.encoder = SentenceTransformer(
            "/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/e5-base-v2"
        )

        self._load_data()

    def _load_data(self):
        """Load corpus and dense index from data directory."""
        print(f"Loading data from {self.data_dir}")

        # Load corpus
        corpus_file = self.data_dir / "../wikipedia/wiki-18.jsonl"
        print(f"Attempting to load corpus from {corpus_file}")
        try:
            # 尝试使用 UTF-8 编码，并忽略无法解码的错误字符
            with open(corpus_file, encoding="utf-8", errors="ignore") as f:
                self.corpus = [json.loads(line) for line in f]
        except UnicodeDecodeError:
            # 如果失败，尝试使用更宽松的编码，例如 Latin-1（它能处理 0x80）
            print("UTF-8 decoding failed. Trying Latin-1 encoding...")
            with open(corpus_file, encoding="latin-1") as f:
                # 注意：使用 Latin-1 可能会导致一些字符被错误解析，
                # 但能避免初始化失败。
                self.corpus = [json.loads(line) for line in f]

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        self.dense_index = faiss.read_index(str(dense_index_file))
        print(f"Loaded dense index with {self.dense_index.ntotal} vectors")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Dense retrieval using FAISS."""
        query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = self.dense_index.search(query_vector, k)

        return [
            {"content": self.corpus[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0], strict=False)
            if idx < len(self.corpus)
        ]


# Pydantic models for request/response validation
class RetrievalRequest(BaseModel):
    """Request model for retrieval endpoint."""

    query: str = Field(..., description="Search query text")
    top_k: int | None = Field(None, description="Number of results to return")
    k: int | None = Field(
        None, description="Alternative parameter for number of results"
    )


class RetrievalResultItem(BaseModel):
    """Single retrieval result."""

    id: str = Field(..., description="Document ID")
    content: dict[str, Any] = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")


class RetrievalResponse(BaseModel):
    """Response model for retrieval endpoint."""

    query: str = Field(..., description="Original query")
    method: str = Field(..., description="Retrieval method used")
    results: list[RetrievalResultItem] = Field(
        ..., description="List of retrieved documents"
    )
    num_results: int = Field(..., description="Total number of results")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    corpus_size: int = Field(..., description="Number of documents in corpus")
    index_type: str = Field(..., description="Type of index used")
    index_loaded: bool = Field(..., description="Whether index is loaded")


# Global retriever instance
retriever: LocalRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup: retriever is already initialized in main()
    yield
    # Shutdown: cleanup if needed
    pass


# FastAPI app
app = FastAPI(
    title="Dense Retrieval Server",
    description="Dense-only retrieval system using E5 embeddings and FAISS indexing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check():
    """Check service health and return system statistics."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    return HealthResponse(
        status="healthy",
        corpus_size=len(retriever.corpus),
        index_type="dense_only",
        index_loaded=retriever.dense_index is not None,
    )


@app.post(
    "/retrieve", response_model=RetrievalResponse, summary="Retrieve relevant documents"
)
async def retrieve(request: RetrievalRequest):
    """
    Perform dense retrieval for the given query.

    Args:
        request: Retrieval request containing query and optional top_k parameter

    Returns:
        Retrieved documents with relevance scores
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Get top_k parameter (support both 'top_k' and 'k')
        k = request.top_k or request.k or 10

        # Perform search
        results = retriever.search(query=request.query, k=k)

        # Format results
        formatted_results = [
            RetrievalResultItem(
                id=f"doc_{i}", content=result["content"], score=result["score"]
            )
            for i, result in enumerate(results, 1)
        ]

        return RetrievalResponse(
            query=request.query,
            method="dense",
            results=formatted_results,
            num_results=len(formatted_results),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Retrieval failed: {str(e)}"
        ) from e


def main():
    parser = argparse.ArgumentParser(description="Dense-only retrieval server")
    parser.add_argument(
        "--data_dir",
        default="mnt/public/sunjinfeng/data/search_data/prebuilt_indices",
        help="Directory containing corpus and dense index",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=2727, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload mode")
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of worker processes"
    )

    args = parser.parse_args()

    start_time = time.time()
    # Initialize retriever
    global retriever
    try:
        retriever = LocalRetriever(args.data_dir)
        print(
            f"Dense retrieval server initialized with {len(retriever.corpus)} documents"
        )
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    # Start server
    elapsed_time = time.time() - start_time
    print(f"Took {elapsed_time:.2f} seconds to initialize the server")
    print(f"Starting dense retrieval server on {args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "retrieval.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
