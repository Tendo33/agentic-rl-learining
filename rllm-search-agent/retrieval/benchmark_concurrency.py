#!/usr/bin/env python3
"""
Benchmark script to test concurrent retrieval performance.

Tests single worker async mode vs multiple worker mode.
Demonstrates that 1 worker + async can handle high concurrency efficiently.

Usage:
    python benchmark_concurrency.py --server http://localhost:2727 --concurrent 100
"""

import argparse
import asyncio
import time
from statistics import mean, median, stdev

import aiohttp


async def single_request(
    session: aiohttp.ClientSession, url: str, query: str, request_id: int
) -> dict:
    """Send a single retrieval request and measure latency."""
    start_time = time.time()
    try:
        async with session.post(
            f"{url}/retrieve",
            json={"query": query, "top_k": 10},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            await response.json()  # Validate response
            latency = time.time() - start_time
            return {
                "id": request_id,
                "success": True,
                "latency": latency,
                "status": response.status,
            }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "id": request_id,
            "success": False,
            "latency": latency,
            "error": str(e),
        }


async def benchmark(
    server_url: str, num_requests: int, concurrency: int, queries: list[str]
) -> dict:
    """
    Run concurrent requests and measure performance.

    Args:
        server_url: Base URL of the server
        num_requests: Total number of requests to send
        concurrency: Number of concurrent requests
        queries: List of queries to cycle through

    Returns:
        Performance statistics
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {server_url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"{'=' * 60}\n")

    start_time = time.time()
    results = []

    # Create connection pool
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create batches of concurrent requests
        for batch_start in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - batch_start)
            tasks = []

            for i in range(batch_size):
                request_id = batch_start + i
                query = queries[request_id % len(queries)]
                task = single_request(session, server_url, query, request_id)
                tasks.append(task)

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Progress update
            completed = len(results)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(
                f"Progress: {completed}/{num_requests} "
                f"({completed * 100 // num_requests}%) - "
                f"{rate:.1f} req/s"
            )

    total_time = time.time() - start_time

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successful]

    stats = {
        "total_requests": num_requests,
        "concurrency": concurrency,
        "total_time": total_time,
        "throughput": num_requests / total_time,
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / num_requests * 100,
    }

    if latencies:
        stats.update(
            {
                "latency_mean": mean(latencies),
                "latency_median": median(latencies),
                "latency_min": min(latencies),
                "latency_max": max(latencies),
                "latency_stdev": stdev(latencies) if len(latencies) > 1 else 0,
            }
        )

    return stats


def print_stats(stats: dict):
    """Print benchmark statistics in a nice format."""
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Total requests:     {stats['total_requests']}")
    print(f"Concurrency:        {stats['concurrency']}")
    print(f"Total time:         {stats['total_time']:.2f}s")
    print(f"Throughput:         {stats['throughput']:.2f} req/s")
    print(f"Success rate:       {stats['success_rate']:.1f}%")
    print(f"Successful:         {stats['successful']}")
    print(f"Failed:             {stats['failed']}")

    if "latency_mean" in stats:
        print("\nLatency Statistics:")
        print(f"  Mean:             {stats['latency_mean'] * 1000:.2f}ms")
        print(f"  Median:           {stats['latency_median'] * 1000:.2f}ms")
        print(f"  Min:              {stats['latency_min'] * 1000:.2f}ms")
        print(f"  Max:              {stats['latency_max'] * 1000:.2f}ms")
        print(f"  Std Dev:          {stats['latency_stdev'] * 1000:.2f}ms")

    print(f"{'=' * 60}\n")


async def check_health(server_url: str) -> bool:
    """Check if server is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Server is healthy")
                    print(f"   Corpus size: {data['corpus_size']}")
                    print(f"   Index type: {data['index_type']}")
                    return True
                else:
                    print(f"❌ Server returned status {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent retrieval performance"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:2727",
        help="Server URL (default: http://localhost:2727)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests (default: 100)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=50,
        help="Number of concurrent requests (default: 50)",
    )

    args = parser.parse_args()

    # Sample queries for testing
    queries = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning",
        "What is artificial intelligence?",
        "Define natural language processing",
        "How to train a model?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What is supervised learning?",
        "Define unsupervised learning",
    ]

    print("\n🚀 Starting Concurrent Retrieval Benchmark")
    print(f"Target server: {args.server}")

    # Check server health
    if not await check_health(args.server):
        print("\n❌ Server is not available. Please start the server first.")
        print("   Example: bash retrieval/launch_server.sh")
        return

    # Run benchmark
    stats = await benchmark(args.server, args.requests, args.concurrent, queries)
    print_stats(stats)

    # Performance assessment
    print("💡 Performance Assessment:")
    if stats["throughput"] > 10:
        print("   ✅ Excellent throughput (>10 req/s)")
    elif stats["throughput"] > 5:
        print("   ⚠️  Good throughput (5-10 req/s)")
    else:
        print("   ❌ Low throughput (<5 req/s)")

    if stats.get("latency_mean", 0) < 0.5:
        print("   ✅ Low latency (<500ms)")
    elif stats.get("latency_mean", 0) < 1.0:
        print("   ⚠️  Moderate latency (500ms-1s)")
    else:
        print("   ❌ High latency (>1s)")

    if stats["success_rate"] >= 99:
        print("   ✅ High reliability (≥99% success)")
    elif stats["success_rate"] >= 95:
        print("   ⚠️  Good reliability (95-99% success)")
    else:
        print("   ❌ Low reliability (<95% success)")


if __name__ == "__main__":
    asyncio.run(main())
