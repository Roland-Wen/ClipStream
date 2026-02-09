import asyncio
import httpx
import time
import numpy as np

URL = "http://localhost:8000/search"
PAYLOAD = {
    "query": "anime character running",
    "top_k": 5
}
CONCURRENT_REQUESTS = 500

async def send_request(client, i):
    start = time.time()
    try:
        resp = await client.post(URL, json=PAYLOAD, timeout=10.0)
        elapsed = (time.time() - start) * 1000
        return elapsed, resp.status_code
    except Exception:
        return 0.0, 500

async def stress_test():
    print(f"🔥 Starting Stress Test: {CONCURRENT_REQUESTS} concurrent requests...")
    
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
        
        start_global = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_global
        
    # Stats
    times = [r[0] for r in results if r[1] == 200]
    errors = [r for r in results if r[1] != 200]
    
    print("\n📊 STRESS TEST RESULTS")
    print("-" * 30)
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Throughput:       {len(times) / total_time:.2f} req/sec")
    print(f"Avg Latency:      {np.mean(times):.2f}ms")
    print(f"P95 Latency:      {np.percentile(times, 95):.2f}ms")
    print(f"Errors:           {len(errors)}")

if __name__ == "__main__":
    asyncio.run(stress_test())