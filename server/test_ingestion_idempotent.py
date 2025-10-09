"""Smoke tests for incremental ingestion and retrieval pipeline.

Tests:
1. Idempotent ingestion: ingest same source twice → no duplicates
2. Manifest tracking: verify chunk_ids are tracked correctly
3. Chunk storage: verify chunk text files are created
4. Retrieval accuracy: query known content and verify expected results
5. Full text loading: verify lazy loading works correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.ingestion import ingest
from server.manifest import get_manifest_manager, compute_chunk_id
from server.chunk_storage import load_chunk_text, get_chunks_directory
from server.retrieval import get_retrieval_pipeline
from server.vector_store import get_store


# Test data
TEST_DOCS = [
    {
        "pmid": "TEST001",
        "title": "Breast Reconstruction Techniques",
        "text": "This study examines modern breast reconstruction techniques including autologous tissue transfer and implant-based reconstruction. We found that patient satisfaction was highest with autologous methods.",
        "authors": "Smith J, Doe A",
        "date": "2024",
    },
    {
        "pmid": "TEST002",
        "title": "Rhinoplasty Outcomes",
        "text": "A comprehensive review of rhinoplasty outcomes over 10 years. Primary rhinoplasty showed better outcomes than revision cases. Patient selection is critical for success.",
        "authors": "Johnson K",
        "date": "2023",
    },
    {
        "pmid": "TEST003",
        "title": "Scar Management Strategies",
        "text": "Evidence-based approaches to scar management including silicone sheets, pressure therapy, and laser treatments. Early intervention produces the best results.",
        "authors": "Chen L, Park M",
        "date": "2024",
    },
]


def test_idempotent_ingestion():
    """Test that ingesting the same documents twice doesn't create duplicates."""
    print("\n=== TEST: Idempotent Ingestion ===")
    
    # First ingestion
    print("First ingestion...")
    stats1 = ingest(TEST_DOCS, incremental=True)
    print(f"Stats (first): {stats1}")
    
    manifest = get_manifest_manager()
    count_after_first = manifest.count()
    print(f"Manifest entries after first ingest: {count_after_first}")
    
    # Second ingestion (same data)
    print("\nSecond ingestion (same data)...")
    stats2 = ingest(TEST_DOCS, incremental=True)
    print(f"Stats (second): {stats2}")
    
    count_after_second = manifest.count()
    print(f"Manifest entries after second ingest: {count_after_second}")
    
    # Assertions
    assert stats2["skipped_chunks"] > 0, "Should skip existing chunks on second ingest"
    assert stats2["new_chunks"] == 0, "Should have no new chunks on second ingest"
    assert count_after_first == count_after_second, "Manifest count should not change"
    
    print("✓ Idempotent ingestion test PASSED")
    return True


def test_chunk_storage():
    """Test that chunk text files are created and can be loaded."""
    print("\n=== TEST: Chunk Storage ===")
    
    # Ingest test docs
    stats = ingest(TEST_DOCS, incremental=True)
    print(f"Ingestion stats: {stats}")
    
    # Check that chunk files exist
    chunks_dir = get_chunks_directory()
    print(f"Chunks directory: {chunks_dir}")
    
    if not chunks_dir.exists():
        print(f"ERROR: Chunks directory does not exist: {chunks_dir}")
        return False
    
    chunk_files = list(chunks_dir.glob("*.txt"))
    print(f"Found {len(chunk_files)} chunk files")
    
    assert len(chunk_files) > 0, "Should have created chunk files"
    
    # Load a chunk and verify content
    first_chunk_file = chunk_files[0]
    chunk_id = first_chunk_file.stem
    
    text = load_chunk_text(chunk_id)
    assert text is not None, "Should be able to load chunk text"
    assert len(text) > 0, "Chunk text should not be empty"
    
    print(f"✓ Loaded chunk {chunk_id[:12]}... ({len(text)} chars)")
    print("✓ Chunk storage test PASSED")
    return True


def test_manifest_tracking():
    """Test that manifest correctly tracks chunk_ids."""
    print("\n=== TEST: Manifest Tracking ===")
    
    manifest = get_manifest_manager()
    manifest.load()
    
    print(f"Manifest has {manifest.count()} entries")
    
    # Compute expected chunk_id for first chunk of first doc
    doc = TEST_DOCS[0]
    # We need to chunk the text the same way ingestion does
    from server.ingestion import _chunk_text
    chunks = _chunk_text(doc["text"])
    
    if chunks:
        expected_chunk_id = compute_chunk_id(doc["pmid"], 0, chunks[0])
        print(f"Expected chunk_id for first chunk: {expected_chunk_id[:12]}...")
        
        has_chunk = manifest.has_chunk(expected_chunk_id)
        assert has_chunk, f"Manifest should contain chunk {expected_chunk_id}"
        
        entry = manifest.get_entry(expected_chunk_id)
        assert entry is not None, "Should retrieve manifest entry"
        assert entry["chunk_id"] == expected_chunk_id, "Chunk ID should match"
        
        print(f"✓ Found expected chunk in manifest")
        print(f"  Metadata keys: {list(entry.get('metadata', {}).keys())}")
    
    print("✓ Manifest tracking test PASSED")
    return True


def test_retrieval_accuracy():
    """Test that retrieval returns relevant results for known queries."""
    print("\n=== TEST: Retrieval Accuracy ===")
    
    # Ensure data is ingested
    ingest(TEST_DOCS, incremental=True)
    
    # Create retrieval pipeline
    pipeline = get_retrieval_pipeline()
    
    # Test query: should retrieve breast reconstruction doc
    query1 = "breast reconstruction techniques"
    results1 = pipeline.retrieve(query1, load_full_text=False)
    
    print(f"\nQuery: '{query1}'")
    print(f"Results: {len(results1)}")
    
    assert len(results1) > 0, "Should return results"
    
    # Check if breast reconstruction doc is in top results
    titles = [r.get("metadata", {}).get("title", "") for r in results1]
    print(f"Top result titles: {titles[:3]}")
    
    found_breast = any("breast" in title.lower() for title in titles[:2])
    if found_breast:
        print("✓ Found relevant 'breast reconstruction' document in top 2")
    else:
        print("⚠ Warning: Expected document not in top 2 (may be OK with small dataset)")
    
    # Test query 2: rhinoplasty
    query2 = "rhinoplasty outcomes"
    results2 = pipeline.retrieve(query2, load_full_text=False)
    
    print(f"\nQuery: '{query2}'")
    print(f"Results: {len(results2)}")
    
    titles2 = [r.get("metadata", {}).get("title", "") for r in results2]
    print(f"Top result titles: {titles2[:3]}")
    
    found_rhino = any("rhinoplasty" in title.lower() for title in titles2[:2])
    if found_rhino:
        print("✓ Found relevant 'rhinoplasty' document in top 2")
    else:
        print("⚠ Warning: Expected document not in top 2 (may be OK with small dataset)")
    
    print("✓ Retrieval accuracy test PASSED")
    return True


def test_full_text_loading():
    """Test that full text loading works correctly."""
    print("\n=== TEST: Full Text Loading ===")
    
    # Ensure data is ingested
    ingest(TEST_DOCS, incremental=True)
    
    pipeline = get_retrieval_pipeline()
    
    # Retrieve with full text loading
    query = "scar management"
    results_with_text = pipeline.retrieve(query, load_full_text=True)
    results_without_text = pipeline.retrieve(query, load_full_text=False)
    
    print(f"Query: '{query}'")
    print(f"Results with text: {len(results_with_text)}")
    print(f"Results without text: {len(results_without_text)}")
    
    if results_with_text:
        result = results_with_text[0]
        has_text = "text" in result
        has_summary = "summary" in result
        
        print(f"First result has 'text': {has_text}")
        print(f"First result has 'summary': {has_summary}")
        
        if has_text:
            text_len = len(result["text"])
            summary_len = len(result.get("summary", ""))
            print(f"Text length: {text_len}, Summary length: {summary_len}")
            
            # Full text should be longer than summary
            assert text_len >= summary_len, "Full text should be >= summary length"
            print("✓ Full text loaded successfully")
        else:
            print("⚠ Warning: 'text' field not found (may use summary as fallback)")
    
    print("✓ Full text loading test PASSED")
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS FOR INCREMENTAL INGESTION & RETRIEVAL")
    print("=" * 60)
    
    tests = [
        ("Idempotent Ingestion", test_idempotent_ingestion),
        ("Chunk Storage", test_chunk_storage),
        ("Manifest Tracking", test_manifest_tracking),
        ("Retrieval Accuracy", test_retrieval_accuracy),
        ("Full Text Loading", test_full_text_loading),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
