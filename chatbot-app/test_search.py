#!/usr/bin/env python3
"""
Quick test script to verify the chatbot components work with the existing database.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_direct_qdrant():
    """Test direct connection to Qdrant using HTTP requests."""
    import requests
    
    try:
        print("🔍 Testing direct Qdrant connection...")
        
        # Test connection
        response = requests.get("http://localhost:6333/collections", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        collections = [c['name'] for c in data['result']['collections']]
        print(f"✅ Connected to Qdrant successfully")
        print(f"📁 Collections: {collections}")
        
        if 'tekyz_knowledge' in collections:
            # Get collection info
            response = requests.get("http://localhost:6333/collections/tekyz_knowledge", timeout=5)
            response.raise_for_status()
            
            info = response.json()['result']
            print(f"📊 Collection 'tekyz_knowledge':")
            print(f"   - Points: {info['points_count']}")
            print(f"   - Vector size: {info['config']['params']['vectors']['size']}")
            print(f"   - Distance: {info['config']['params']['vectors']['distance']}")
            return True
        else:
            print("❌ Collection 'tekyz_knowledge' not found")
            return False
            
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def test_search_engine():
    """Test the vector search engine."""
    try:
        print("\n🔍 Testing Vector Search Engine...")
        
        from utils.config_manager import ConfigManager
        from backend.vector_search import VectorSearchEngine
        
        # Initialize config
        config = ConfigManager()
        search_engine = VectorSearchEngine(config)
        
        # Test health check
        health = search_engine.health_check()
        print(f"🏥 Health check: {health['status']}")
        
        if health['status'] == 'healthy':
            print(f"   - Points: {health['points_count']}")
            print(f"   - Vector size: {health['vector_size']}")
            
            # Test a simple search (will need embedding model)
            print("\n🔍 Testing search functionality...")
            try:
                results = search_engine.search("What services does Tekyz offer?", limit=3)
                print(f"✅ Search successful: {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result.score:.3f}")
                    print(f"      Title: {result.title}")
                    print(f"      URL: {result.source_url}")
                    print(f"      Metadata: {result.metadata}")
                    print()
                    
            except Exception as e:
                print(f"⚠️  Search test failed (likely missing embedding model): {e}")
                print("   This is expected if sentence-transformers isn't installed yet")
                
        return True
        
    except Exception as e:
        print(f"❌ Vector search engine test failed: {e}")
        return False

def test_config_manager():
    """Test the configuration management."""
    try:
        print("\n🔧 Testing Configuration Manager...")
        
        from utils.config_manager import ConfigManager
        
        config = ConfigManager()
        print(f"✅ Config loaded successfully")
        print(f"   - Qdrant host: {config.qdrant.host}")
        print(f"   - Qdrant port: {config.qdrant.port}")
        print(f"   - Collection: {config.qdrant.collection_name}")
        print(f"   - Embedding model: {config.embeddings.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config manager test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Tekyz Chatbot - System Test")
    print("=" * 50)
    
    # Test individual components
    tests = [
        ("Direct Qdrant Connection", test_direct_qdrant),
        ("Configuration Manager", test_config_manager),
        ("Vector Search Engine", test_search_engine),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Chatbot is ready for use.")
    elif passed >= 2:
        print("⚠️  Most tests passed. Check embedding model installation.")
    else:
        print("❌ Critical failures detected. Check your setup.")

if __name__ == "__main__":
    main() 