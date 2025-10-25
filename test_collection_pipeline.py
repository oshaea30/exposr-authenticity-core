"""
Test script for the public domain collection pipeline
"""
import os
import sys
import subprocess
from pathlib import Path

def test_collection_pipeline():
    """Test the collection pipeline with small sample"""
    print("🧪 Testing Public Domain Collection Pipeline")
    print("=" * 50)
    
    # Test real image collection (small sample)
    print("\n📥 Testing real image collection (5 images)...")
    try:
        result = subprocess.run([
            sys.executable, "scrape_public_real_images.py", 
            "--count", "5", "--wikimedia-only"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Real image collection test passed")
        else:
            print(f"❌ Real image collection test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Real image collection test error: {e}")
        return False
    
    # Test AI image collection (small sample)
    print("\n🤖 Testing AI image collection (5 images)...")
    try:
        result = subprocess.run([
            sys.executable, "scrape_public_ai_images.py", 
            "--count", "5", "--lexica-only"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ AI image collection test passed")
        else:
            print(f"❌ AI image collection test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ AI image collection test error: {e}")
        return False
    
    # Test embedding generation
    print("\n🧠 Testing embedding generation...")
    try:
        result = subprocess.run([
            sys.executable, "embed_images.py", 
            "--version", "test"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Embedding generation test passed")
        else:
            print(f"❌ Embedding generation test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Embedding generation test error: {e}")
        return False
    
    # Test master pipeline
    print("\n🚀 Testing master pipeline...")
    try:
        result = subprocess.run([
            sys.executable, "collect_and_embed.py", 
            "--ai-count", "2", "--real-count", "2", "--version", "test"
        ], capture_output=True, text=True, timeout=900)
        
        if result.returncode == 0:
            print("✅ Master pipeline test passed")
        else:
            print(f"❌ Master pipeline test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Master pipeline test error: {e}")
        return False
    
    # Check results
    print("\n📊 Checking results...")
    ai_count = len(list(Path("data/ai").glob("*.jpg")))
    real_count = len(list(Path("data/real").glob("*.jpg")))
    embeddings_exist = Path("embeddings/latest_embeddings.npy").exists()
    metadata_exists = Path("dataset_log.csv").exists()
    
    print(f"  • AI images collected: {ai_count}")
    print(f"  • Real images collected: {real_count}")
    print(f"  • Embeddings generated: {embeddings_exist}")
    print(f"  • Metadata logged: {metadata_exists}")
    
    if ai_count > 0 and real_count > 0 and embeddings_exist and metadata_exists:
        print("\n🎉 All tests passed! Pipeline is ready for production use.")
        return True
    else:
        print("\n❌ Some tests failed. Check the results above.")
        return False

if __name__ == "__main__":
    success = test_collection_pipeline()
    sys.exit(0 if success else 1)
