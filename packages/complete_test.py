#!/usr/bin/env python3
"""
Complete test script for packages and memory system
Includes all the individual package tests plus memory system integration
"""

import sys
import os
import importlib

def test_numpy_import():
    """Test numpy import (fixed formatting)"""
    print("🔍 Testing NumPy import...")
    try:
        import numpy
        print(f'✅ NumPy imported successfully! Version: {numpy.__version__}')
        return True
    except Exception as e:
        print(f'❌ NumPy import failed: {e}')
        return False

def test_sklearn_import():
    """Test sklearn import"""
    print("🔍 Testing Scikit-learn import...")
    try:
        import sklearn
        print(f'✅ Scikit-learn imported successfully! Version: {sklearn.__version__}')
        return True
    except Exception as e:
        print(f'❌ Scikit-learn import failed: {e}')
        return False

def test_all_packages():
    """Test all packages at once"""
    print("🔍 Testing all packages...")
    packages = ['numpy', 'scipy', 'sklearn', 'joblib', 'threadpoolctl']
    success_count = 0
    
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f'✅ {pkg}: {version}')
            success_count += 1
        except ImportError as e:
            print(f'❌ {pkg}: {e}')
    
    return success_count

def test_enhanced_functionality():
    """Test enhanced functionality that requires numpy/sklearn"""
    print("\n🧠 Testing enhanced functionality...")
    print("=" * 50)
    
    # Test numpy functionality
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = arr.mean()
        print(f"✅ NumPy array operations working: mean = {mean_val}")
    except Exception as e:
        print(f"❌ NumPy operations failed: {e}")
        return False
    
    # Test sklearn TfidfVectorizer
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        test_texts = ["This is a test document", "Another test document for analysis"]
        tfidf_matrix = vectorizer.fit_transform(test_texts)
        print(f"✅ TfidfVectorizer working: matrix shape {tfidf_matrix.shape}")
    except Exception as e:
        print(f"❌ TfidfVectorizer failed: {e}")
        return False
    
    # Test cosine similarity
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(tfidf_matrix)
        print(f"✅ Cosine similarity working: similarity matrix {similarity.shape}")
        print(f"   Sample similarity score: {similarity[0][1]:.3f}")
    except Exception as e:
        print(f"❌ Cosine similarity failed: {e}")
        return False
    
    print("🎉 Enhanced functionality fully operational!")
    return True

def test_memory_system_integration():
    """Test memory system with enhanced capabilities"""
    print("\n🚀 Testing Memory System Integration...")
    print("=" * 50)
    
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
        print("✅ SemanticAnalyzer imported successfully")
        
        # Initialize analyzer
        analyzer = SemanticAnalyzer(prsist_dir, parent_dir)
        
        # Check if enhanced features are available
        sklearn_available = hasattr(analyzer, 'tfidf_vectorizer') and analyzer.tfidf_vectorizer is not None
        numpy_available = hasattr(analyzer, 'SKLEARN_AVAILABLE') and analyzer.SKLEARN_AVAILABLE
        
        print(f"✅ TF-IDF vectorizer available: {sklearn_available}")
        print(f"✅ Sklearn integration active: {numpy_available}")
        
        if sklearn_available:
            print("🎉 Enhanced semantic analysis with TF-IDF is ready!")
            
            # Test embedding generation with enhanced features
            test_text = "This is a test of the enhanced memory system semantic analysis capabilities"
            embedding = analyzer._generate_text_embedding(test_text, "test_enhanced", "test")
            
            if embedding:
                method = embedding.metadata.get('generated_method', 'unknown')
                vector_size = len(embedding.embedding_vector)
                print(f"✅ Enhanced embedding generated successfully!")
                print(f"   Method: {method}")
                print(f"   Vector size: {vector_size}")
                
                # Test similarity calculation if we have numpy
                try:
                    embedding2 = analyzer._generate_text_embedding("Another test text for similarity", "test2", "test")
                    if embedding2:
                        similarity = analyzer.calculate_similarity(embedding, embedding2)
                        print(f"✅ Similarity calculation working: {similarity:.3f}")
                except Exception as e:
                    print(f"🟡 Similarity calculation issue: {e}")
            else:
                print("❌ Embedding generation failed")
        else:
            print("🟡 Using fallback semantic analysis (TF-IDF not available)")
            
        # Test file analysis
        memory_manager_path = os.path.join(prsist_dir, 'memory_manager.py')
        if os.path.exists(memory_manager_path):
            analysis = analyzer.analyze_file_semantics(memory_manager_path)
            if 'error' not in analysis:
                language = analysis.get('language', 'unknown')
                elements = len(analysis.get('elements', []))
                keywords = len(analysis.get('semantic_keywords', []))
                print(f"✅ File analysis working: {language}, {elements} elements, {keywords} keywords")
            else:
                print(f"❌ File analysis failed: {analysis.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pip_functionality():
    """Test that pip is working"""
    print("\n🔧 Testing pip functionality...")
    print("=" * 30)
    
    try:
        import pip
        print(f"✅ Pip module imported: {pip.__version__}")
        
        # Test pip command
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Pip command working: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Pip command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Pip test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 Complete Enhanced Package and Memory System Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Test individual imports first
    print("Phase 1: Individual Package Tests")
    print("-" * 40)
    numpy_ok = test_numpy_import()
    sklearn_ok = test_sklearn_import()
    
    print("\nPhase 2: All Packages Test")
    print("-" * 40)
    package_count = test_all_packages()
    
    print(f"\nPackage Import Summary: {package_count}/5 packages working")
    
    # Test pip functionality
    pip_ok = test_pip_functionality()
    
    # Only proceed with advanced tests if basic imports work
    enhanced_ok = False
    memory_ok = False
    
    if numpy_ok and sklearn_ok:
        print("\nPhase 3: Enhanced Functionality Tests")
        print("-" * 40)
        enhanced_ok = test_enhanced_functionality()
        
        print("\nPhase 4: Memory System Integration")
        print("-" * 40)
        memory_ok = test_memory_system_integration()
    else:
        print("\n🟡 Skipping advanced tests due to import failures")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Final Test Results Summary")
    print("=" * 60)
    
    results = {
        "NumPy Import": numpy_ok,
        "Scikit-learn Import": sklearn_ok, 
        "Package Count": f"{package_count}/5",
        "Pip Functionality": pip_ok,
        "Enhanced Features": enhanced_ok,
        "Memory System": memory_ok
    }
    
    for test_name, result in results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
        else:
            status = f"📊 {result}"
        print(f"{test_name:20}: {status}")
    
    overall_success = numpy_ok and sklearn_ok and package_count >= 3
    
    if overall_success and enhanced_ok and memory_ok:
        print("\n🎉 COMPLETE SUCCESS!")
        print("✅ All packages installed and working correctly")
        print("✅ Enhanced semantic analysis with TF-IDF active")
        print("✅ Memory system integration successful")
        print("✅ Advanced similarity calculations enabled")
        print("\n🚀 Next Steps:")
        print("   • Run: cd .. && py -3.13 .prsist/test_runner.py")
        print("   • Your enhanced Prsist Memory System is ready!")
        return True
        
    elif overall_success:
        print("\n🟡 PARTIAL SUCCESS!")
        print("✅ Core packages working")
        print("🟡 Some advanced features may not be available")
        print("\n🚀 You can still use the memory system with basic functionality")
        return True
        
    else:
        print("\n❌ INSTALLATION ISSUES DETECTED")
        print("❌ Core packages not working properly") 
        print("🔧 Check error messages above for troubleshooting")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n💡 Script completed with {'SUCCESS' if success else 'ISSUES'}")
    sys.exit(0 if success else 1)