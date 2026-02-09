#!/bin/bash
for i in {1..25}
do
   echo "Request $i..."
   curl -s -o /dev/null -w "%{http_code}\n" \
     -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "top_k": 1}'
done