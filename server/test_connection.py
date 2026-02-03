"""Quick test script to verify the app is working."""

import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_google_ai():
    """Test Google AI connection."""
    print("=" * 50)
    print("Testing Google AI Configuration")
    print("=" * 50)
    
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    
    api_key = os.getenv("GOOGLE_API_KEY", "")
    print(f"✓ API Key set: {bool(api_key)}")
    print(f"  Key prefix: {api_key[:10]}..." if api_key else "  No key!")
    
    # Test import
    try:
        from google import genai
        print("✓ google-genai package installed")
    except ImportError:
        print("✗ google-genai NOT installed. Run: pip install google-genai")
        return False
    
    # Test client creation
    try:
        client = genai.Client(api_key=api_key)
        print("✓ Client created successfully")
    except Exception as e:
        print(f"✗ Client creation failed: {e}")
        return False
    
    # Test embedding
    print("\nTesting embeddings...")
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents="test embedding"
        )
        print(f"✓ Embedding works! Vector dimension: {len(result.embeddings[0].values)}")
    except Exception as e:
        print(f"✗ Embedding failed: {e}")
        return False
    
    # Test chat
    print("\nTesting chat generation...")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Hello, the API is working!' in exactly those words.",
        )
        print(f"✓ Chat works! Response: {response.text[:100]}")
    except Exception as e:
        print(f"✗ Chat failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_google_ai()
    sys.exit(0 if success else 1)
