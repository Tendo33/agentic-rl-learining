#!/usr/bin/env python3
"""
Dense-only retrieval server for Search training.
Provides E5 embeddings + FAISS dense indexing.

Usage:
    python server.py --data_dir ./search_data/prebuilt_indices --port 8000
"""

import argparse
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        logger.info(f"Loading data from {self.data_dir}")

        # Load corpus
        corpus_file = self.data_dir / "../wikipedia/wiki-18.jsonl"
        logger.info(f"Loading corpus from {corpus_file}")

        start_time = time.time()
        try:
            # 尝试使用 UTF-8 编码，并忽略无法解码的错误字符
            with open(corpus_file, encoding="utf-8", errors="ignore") as f:
                self.corpus = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.corpus)} documents from corpus")
        except UnicodeDecodeError:
            # 如果失败，尝试使用更宽松的编码，例如 Latin-1（它能处理 0x80）
            logger.warning("UTF-8 decoding failed. Trying Latin-1 encoding...")
            with open(corpus_file, encoding="latin-1") as f:
                self.corpus = [json.loads(line) for line in f]
            logger.info(f"Loaded {len(self.corpus)} documents from corpus (Latin-1)")

        corpus_load_time = time.time() - start_time
        logger.info(f"Corpus loading took {corpus_load_time:.2f} seconds")

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        logger.info(f"Loading dense index from {dense_index_file}")

        index_start = time.time()
        self.dense_index = faiss.read_index(str(dense_index_file))
        index_load_time = time.time() - index_start

        logger.info(
            f"Loaded dense index with {self.dense_index.ntotal} vectors "
            f"in {index_load_time:.2f} seconds"
        )

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


# Global retriever instance and config
retriever: LocalRetriever | None = None
config = {
    "data_dir": "/mnt/public/sunjinfeng/data/search_data/prebuilt_indices",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI app.

    Handles initialization and cleanup of the retriever instance.
    This ensures the retriever is loaded once per worker process.
    """
    global retriever

    # Startup: Initialize retriever
    logger.info("=" * 60)
    logger.info("Starting FastAPI Dense Retrieval Server")
    logger.info("=" * 60)

    if config["data_dir"] is None:
        logger.error("Data directory not configured!")
        raise ValueError("Data directory must be set before starting server")

    start_time = time.time()
    try:
        logger.info(f"Initializing retriever with data_dir: {config['data_dir']}")
        retriever = LocalRetriever(config["data_dir"])
        elapsed = time.time() - start_time
        logger.info(f"✓ Retriever initialized successfully in {elapsed:.2f} seconds")
        logger.info(f"✓ Loaded {len(retriever.corpus)} documents")
        logger.info(f"✓ Dense index size: {retriever.dense_index.ntotal} vectors")
    except Exception as e:
        logger.error(f"✗ Failed to initialize retriever: {e}")
        raise

    logger.info("=" * 60)
    logger.info("Server ready to accept requests")
    logger.info("=" * 60)

    yield

    # Shutdown: cleanup resources
    logger.info("Shutting down retriever...")
    retriever = None
    logger.info("Retriever shutdown complete")


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
    """
    Main entry point for the retrieval server.

    Configures and starts the FastAPI server with proper lifespan management.
    """
    parser = argparse.ArgumentParser(
        description="Dense-only retrieval server with FastAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings
  python server.py
  
  # Custom data directory and port
  python server.py --data_dir /path/to/data --port 8080
  
  # Production mode with multiple workers
  python server.py --workers 4 --host 0.0.0.0
  
  # Development mode with auto-reload (single worker only)
  python server.py --reload --workers 1
        """,
    )
    parser.add_argument(
        "--data_dir",
        default="/mnt/public/sunjinfeng/data/search_data/prebuilt_indices",
        help="Directory containing corpus and dense index",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for external access)",
    )
    parser.add_argument("--port", type=int, default=2727, help="Port to bind to")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload mode (development only, forces workers=1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (note: each worker loads the model separately)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.reload and args.workers > 1:
        logger.warning("Auto-reload mode requires single worker. Setting workers=1")
        args.workers = 1

    # Set configuration for lifespan
    config["data_dir"] = args.data_dir

    # Log startup info
    logger.info("=" * 60)
    logger.info("Dense Retrieval Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload mode: {args.reload}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 60)

    if args.workers > 1:
        logger.warning(
            f"⚠️  Running with {args.workers} workers. "
            "Each worker will load its own copy of the model and index!"
        )
        logger.warning("⚠️  Consider reducing workers to save memory if not needed.")

    # Start server (lifespan will handle retriever initialization)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"📚 API documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"🏥 Health check: http://{args.host}:{args.port}/health")
    logger.info("=" * 60)

    uvicorn.run(
        "retrieval.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
