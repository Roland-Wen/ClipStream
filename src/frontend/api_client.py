import httpx
from typing import Dict, Any, Optional, List
import streamlit as st

class ClipStreamClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.timeout = 10.0

    def search(
        self, 
        query: str, 
        top_k: int = 10,  # Increased default to allow for client-side filtering
        categories: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        sort_by: str = "confidence"
    ) -> Dict[str, Any]:
        """
        Sends a search request with metadata filters and sorting options.
        """
        # 'sort_by' is a Query Parameter in the backend
        url = f"{self.base_url}/search"
        params = {"sort_by": sort_by}
        
        # 'categories' and 'years' are Body Parameters (SearchRequest schema)
        payload = {
            "query": query,
            "top_k": top_k,
            "categories": categories or [],
            "years": years or []
        }

        try:
            response = httpx.post(url, params=params, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                return {"error": True, "message": "⚠️ Rate limit exceeded. Please wait."}
            elif response.status_code == 404:
                 return {"error": True, "message": "No matching videos found."}
            else:
                return {"error": True, "message": f"Server Error ({response.status_code}): {response.text}"}

        except httpx.ConnectError:
            return {"error": True, "message": "❌ Could not connect to backend."}
        except httpx.TimeoutException:
            return {"error": True, "message": "⏱️ Request timed out."}
        except Exception as e:
            return {"error": True, "message": f"Unexpected error: {str(e)}"}