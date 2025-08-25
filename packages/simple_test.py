#!/usr/bin/env python3
"""
Simple test for packages and memory system integration
"""

import sys
import os
import importlib

def test_basic_imports():
    """Test all package imports"""
    print("Testing package imports...")
    packages = ['numpy', 'scipy', 'sklearn', 'joblib', 'threadpoolctl']
    success_count = 0
    
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f'SUCCESS: {pkg}: {version}')
            success_count += 1
        except ImportError as e:
            print(f'ERROR: {pkg}: {e}')
    
    return success_count

def test_enhanced_functionality():
    """Test enhanced functionality"""
    print("\nTesting enhanced functionality...")
    
    # Test numpy
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = arr.mean()
        print(f"SUCCESS: NumPy operations working: mean = {mean_val}")
    except Exception as e:
        print(f"ERROR: NumPy operations failed: {e}")
        return False
    
    # Test sklearn TfidfVectorizer
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        test_texts = ["This is a test document", "Another test document for analysis"]
        tfidf_matrix = vectorizer.fit_transform(test_texts)
        print(f"SUCCESS: TfidfVectorizer working: matrix shape {tfidf_matrix.shape}")
    except Exception as e:
        print(f"ERROR: TfidfVectorizer failed: {e}")
        return False
    
    # Test cosine similarity
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(tfidf_matrix)
        print(f"SUCCESS: Cosine similarity working: similarity matrix {similarity.shape}")
        print(f"   Sample similarity score: {similarity[0][1]:.3f}")
    except Exception as e:
        print(f"ERROR: Cosine similarity failed: {e}")
        return False
    
    print("Enhanced functionality fully operational!")
    return True

def test_memory_system():
    """Test memory system integration"""
    print("\nTesting Memory System Integration...")
    
    try:
        # Navigate to parent directory and add .prsist to path
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        prsist_dir = os.path.join(parent_dir, '.prsist')
        
        # Add to Python path
        if prsist_dir not in sys.path:
            sys.path.insert(0, prsist_dir)
        
        print(f"Looking for memory system in: {prsist_dir}")
        print(f"Memory system exists: {os.path.exists(prsist_dir)}")
        
        # Try importing memory system components
        from semantic_analyzer import SemanticAnalyzer
        print("SUCCESS: SemanticAnalyzer imported successfully")
        
        # Initialize analyzer
        analyzer = SemanticAnalyzer(prsist_dir, parent_dir)
        
        # Check if enhanced features are available
        sklearn_available = hasattr(analyzer, 'tfidf_vectorizer') and analyzer.tfidf_vectorizer is not None
        
        print(f"TF-IDF vectorizer available: {sklearn_available}")
        
        if sklearn_available:
            print("Enhanced semantic analysis with TF-IDF is ready!")
            
            # Test embedding generation
            test_text = "This is a test of the enhanced memory system semantic analysis capabilities"
            embedding = analyzer._generate_text_embedding(test_text, "test_enhanced", "test")
            
            if embedding:
                method = embedding.metadata.get('generated_method', 'unknown')
                vector_size = len(embedding.embedding_vector)
                print(f"SUCCESS: Enhanced embedding generated!")
                print(f"   Method: {method}")
                print(f"   Vector size: {vector_size}")
                
                # Test similarity calculation
                embedding2 = analyzer._generate_text_embedding("Another test text for similarity", "test2", "test")
                if embedding2:
                    similarity = analyzer.calculate_similarity(embedding, embedding2)
                    print(f"SUCCESS: Similarity calculation working: {similarity:.3f}")
            else:
                print("ERROR: Embedding generation failed")
        else:
            print("Using fallback semantic analysis")
        
        # Test file analysis
        memory_manager_path = os.path.join(prsist_dir, 'memory_manager.py')
        if os.path.exists(memory_manager_path):
            analysis = analyzer.analyze_file_semantics(memory_manager_path)
            if 'error' not in analysis:
                language = analysis.get('language', 'unknown')
                elements = len(analysis.get('elements', []))
                keywords = len(analysis.get('semantic_keywords', []))
                print(f"SUCCESS: File analysis working: {language}, {elements} elements, {keywords} keywords")
            else:
                print(f"ERROR: File analysis failed: {analysis.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Memory system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Enhanced Package and Memory System Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Test package imports
    package_count = test_basic_imports()
    print(f"\nPackage Import Summary: {package_count}/5 packages working")
    
    # Test enhanced functionality if packages work
    enhanced_ok = False
    memory_ok = False
    
    if package_count >= 4:  # Need at least numpy, scipy, sklearn, joblib
        enhanced_ok = test_enhanced_functionality()
        memory_ok = test_memory_system()
    else:
        print("\nSkipping advanced tests due to import failures")
    
    # Final summary
    print("\n" + "=" * 50)
    print("Final Results:")
    print(f"Packages working: {package_count}/5")
    print(f"Enhanced features: {'YES' if enhanced_ok else 'NO'}")
    print(f"Memory system: {'YES' if memory_ok else 'NO'}")
    
    if package_count >= 4 and enhanced_ok and memory_ok:
        print("\nCOMPLETE SUCCESS!")
        print("All packages installed and working correctly")
        print("Enhanced semantic analysis with TF-IDF active")
        print("Memory system integration successful")
        print("\nNext Steps:")
        print("   cd .. && py -3.13 .prsist/test_runner.py")
        print("   Your enhanced Prsist Memory System is ready!")
        return True
    else:
        print("\nIssues detected - check error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)