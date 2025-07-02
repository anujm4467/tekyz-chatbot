#!/usr/bin/env python3
"""
Test script to diagnose vector search issues
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add src path for imports
sys.path.append('src')

def test_vector_database_direct():
    """Test the vector database directly via HTTP API"""
    try:
        print("🔍 Testing Vector Database Direct Access...")
        
        # Test Qdrant connection
        qdrant_url = "http://localhost:6333"
        response = requests.get(f"{qdrant_url}/collections", timeout=5)
        
        if response.status_code == 200:
            collections = response.json()
            print(f"✅ Qdrant connected. Collections: {[c['name'] for c in collections.get('result', {}).get('collections', [])]}")
            
            # Check tekyz_knowledge collection
            if 'tekyz_knowledge' in [c['name'] for c in collections.get('result', {}).get('collections', [])]:
                collection_info = requests.get(f"{qdrant_url}/collections/tekyz_knowledge").json()
                points_count = collection_info['result']['points_count']
                print(f"✅ tekyz_knowledge collection has {points_count} vectors")
                
                # Get some sample points to see what's stored
                sample_response = requests.post(
                    f"{qdrant_url}/collections/tekyz_knowledge/points/scroll",
                    json={"limit": 3, "with_payload": True}
                )
                
                if sample_response.status_code == 200:
                    points = sample_response.json()['result']['points']
                    print(f"\n📄 Sample stored data:")
                    for i, point in enumerate(points):
                        content = point['payload'].get('content', 'No content')[:200]
                        source = point['payload'].get('source', 'Unknown source')
                        print(f"   {i+1}. Source: {source}")
                        print(f"      Content: {content}...")
                        
                        # Check if this point contains contact info
                        if 'contact' in content.lower() or 'email' in content.lower() or 'phone' in content.lower() or 'tekyz.com' in content.lower():
                            print(f"      🎯 Contains potential contact info!")
                        print()
                
            else:
                print("❌ tekyz_knowledge collection not found")
        else:
            print(f"❌ Qdrant connection failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")

def test_chatbot_backend():
    """Test the chatbot backend health and search"""
    try:
        print("\n🤖 Testing Chatbot Backend...")
        
        # Test health endpoint
        health_response = requests.get("http://localhost:8080/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print("✅ Chatbot backend is running")
            print(f"   Vector DB status: {health_data.get('vector_db', {}).get('status', 'unknown')}")
            print(f"   Ready for search: {health_data.get('vector_db', {}).get('ready_for_search', False)}")
            
            # Test direct search
            search_response = requests.post(
                "http://localhost:8080/search",
                json={"query": "How can I contact Tekyz? Email phone contact information"},
                timeout=10
            )
            
            if search_response.status_code == 200:
                search_results = search_response.json()
                print(f"\n🔍 Search Results for 'contact Tekyz':")
                print(f"   Found {len(search_results.get('results', []))} results")
                
                for i, result in enumerate(search_results.get('results', [])[:3]):
                    print(f"\n   Result {i+1}:")
                    print(f"   Score: {result.get('score', 0):.3f}")
                    print(f"   Source: {result.get('source', 'Unknown')}")
                    content = result.get('content', 'No content')[:300]
                    print(f"   Content: {content}...")
                    
                    # Check if this result contains contact info
                    if any(term in content.lower() for term in ['contact', 'email', 'phone', 'tekyz.com', '@']):
                        print(f"   🎯 Contains contact information!")
            else:
                print(f"❌ Search failed: {search_response.status_code}")
                
        else:
            print(f"❌ Chatbot backend not accessible: {health_response.status_code}")
            
    except Exception as e:
        print(f"❌ Chatbot backend test failed: {e}")

def test_embedding_similarity():
    """Test if we can load the embedding model and check similarity"""
    try:
        print("\n🧠 Testing Embedding Model...")
        
        from sentence_transformers import SentenceTransformer
        
        # Load the same model the chatbot uses
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Test queries
        queries = [
            "How can I contact Tekyz?",
            "Tekyz contact information",
            "email phone Tekyz",
            "info@tekyz.com"
        ]
        
        # Test documents (simulate what might be in the vector DB)
        documents = [
            "Email and phone number info@tekyz.com and (480) 570-8557",
            "Company Website - Tekyz contact details",
            "For inquiries, reach out to Tekyz at info@tekyz.com"
        ]
        
        print("\n📊 Similarity Test:")
        query_embeddings = model.encode(queries)
        doc_embeddings = model.encode(documents)
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        for i, query in enumerate(queries):
            similarities = cosine_similarity([query_embeddings[i]], doc_embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]
            
            print(f"\n   Query: '{query}'")
            print(f"   Best match: '{documents[best_match_idx]}'")
            print(f"   Similarity: {best_score:.3f}")
            
            if best_score > 0.3:  # Typical threshold
                print(f"   ✅ Should be found (score > 0.3)")
            else:
                print(f"   ⚠️ Might be missed (score ≤ 0.3)")
                
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")

if __name__ == "__main__":
    print("🧪 Vector Search Diagnostic Test")
    print("=" * 50)
    
    test_vector_database_direct()
    test_chatbot_backend() 
    test_embedding_similarity()
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic test complete!") 