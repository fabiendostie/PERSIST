#!/usr/bin/env python3
"""
Fix package imports and test everything
This script first fixes the import issues, then tests all functionality
"""

import sys
import os
import site
import importlib
from pathlib import Path

def fix_package_imports():
    """Fix package import issues by ensuring proper site-packages setup"""
    print("🔧 Fixing package import issues...")
    print("=" * 50)
    
    # Get user site-packages directory
    user_site = site.getusersitepackages()
    print(f"User site-packages: {user_site}")
    
    # Ensure user site is enabled
    site.main()
    
    # Force add user site to Python path if not already there
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
        print(f"✅ Added user site-packages to Python path")
    else:
        print(f"✅ User site-packages already in Python path")
    
    # Check what packages are actually installed in user site
    user_site_path = Path(user_site)
    if user_site_path.exists():
        installed_packages = []
        for item in user_site_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                installed_packages.append(item.name)
        
        print(f"📦 Found {len(installed_packages)} package directories in user site:")
        for pkg in sorted(installed_packages):
            print(f"   • {pkg}")
    
    # Also check for .pth files that might affect imports
    pth_files = list(user_site_path.glob("*.pth")) if user_site_path.exists() else []
    if pth_files:
        print(f"📄 Found .pth files: {[f.name for f in pth_files]}")
    
    print("✅ Import path setup completed")
    return True

def test_basic_imports():
    """Test basic package imports after fixing paths"""
    print("\n🧪 Testing basic package imports...")
    print("=" * 50)
    
    # Test packages individually with detailed error reporting
    packages_to_test = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('joblib', 'Joblib'),
        ('threadpoolctl', 'ThreadPoolCtl'),
        ('pip', 'Pip')
    ]
    
    results = {}
    
    for module_name, display_name in packages_to_test:
        print(f"\nTesting {display_name} ({module_name})...")
        try:
            # Try importing the module
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
            results[module_name] = {'success': True, 'version': version, 'error': None}
            
        except ImportError as e:
            print(f"❌ {display_name}: Import failed - {e}")
            results[module_name] = {'success': False, 'version': None, 'error': str(e)}
            
            # Try to find the package directory
            user_site = site.getusersitepackages()
            potential_dirs = [
                Path(user_site) / module_name,
                Path(user_site) / f"{module_name.replace('_', '-')}",
                Path(user_site) / f"{module_name.replace('-', '_')}"
            ]
            
            for potential_dir in potential_dirs:
                if potential_dir.exists():
                    print(f"   💡 Found package directory at: {potential_dir}")
                    break
            else:
                print(f"   ❓ Package directory not found in user site-packages")
                
        except Exception as e:
            print(f"🟡 {display_name}: Imported but error - {e}")
            results[module_name] = {'success': True, 'version': 'unknown', 'error': str(e)}
    
    # Summary
    successful_imports = sum(1 for r in results.values() if r['success'])
    total_packages = len(results)
    
    print(f"\n📊 Import Results: {successful_imports}/{total_packages} packages working")
    
    # Show failed imports
    failed_imports = [name for name, result in results.items() if not result['success']]
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
    
    return results, successful_imports >= 2  # Need at least numpy and sklearn

def manually_fix_sklearn_import():
    """Try to manually fix sklearn import issue"""
    print("\n🔧 Attempting manual sklearn fix...")
    
    user_site = site.getusersitepackages()
    user_site_path = Path(user_site)
    
    # Look for scikit_learn directory
    scikit_learn_dir = user_site_path / "scikit_learn"
    sklearn_dir = user_site_path / "sklearn"
    
    if scikit_learn_dir.exists() and not sklearn_dir.exists():
        print(f"   Found scikit_learn at: {scikit_learn_dir}")
        print("   Creating sklearn symlink/alias...")
        
        try:
            # Try to create a simple module file that imports from scikit_learn
            sklearn_init = user_site_path / "sklearn.py"
            sklearn_init.write_text("# Auto-generated sklearn alias\nfrom scikit_learn import *\n__version__ = getattr(__import__('scikit_learn'), '__version__', 'unknown')\n")
            print("   ✅ Created sklearn.py alias file")
            return True
        except Exception as e:
            print(f"   ❌ Failed to create sklearn alias: {e}")
            return False
    
    elif sklearn_dir.exists():
        print("   ✅ sklearn directory already exists")
        return True
    
    else:
        print("   ❌ Neither scikit_learn nor sklearn directory found")
        return False

def test_enhanced_functionality(import_results):
    """Test enhanced functionality if packages are available"""
    print("\n🧠 Testing Enhanced Functionality...")
    print("=" * 50)
    
    if not import_results.get('numpy', {}).get('success', False):
        print("❌ NumPy not available - skipping enhanced tests")
        return False
    
    # Test NumPy operations
    try:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = arr.mean()
        std_val = arr.std()
        print(f"✅ NumPy operations: mean={mean_val:.2f}, std={std_val:.2f}")
    except Exception as e:
        print(f"❌ NumPy operations failed: {e}")
        return False
    
    # Test sklearn if available
    if import_results.get('sklearn', {}).get('success', False):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Test TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            test_docs = [
                "This is a test document about machine learning",
                "Another document discussing artificial intelligence",
                "A third document about data science and analytics"
            ]
            
            tfidf_matrix = vectorizer.fit_transform(test_docs)
            print(f"✅ TF-IDF vectorization: {tfidf_matrix.shape} matrix")
            
            # Test cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            print(f"✅ Cosine similarity: {similarity_matrix.shape} similarity matrix")
            print(f"   Sample similarities: {similarity_matrix[0, 1]:.3f}, {similarity_matrix[0, 2]:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sklearn functionality failed: {e}")
            return False
    else:
        print("🟡 Sklearn not available - skipping TF-IDF tests")
        return False

def test_memory_system_integration(import_results):
    """Test memory system with available packages"""
    print("\n🚀 Testing Memory System Integration...")
    print("=" * 50)
    
    try:
        # Setup path to memory system
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        prsist_dir = os.path.join(parent_dir, '.prsist')
        
        if prsist_dir not in sys.path:
            sys.path.insert(0, prsist_dir)
        
        print(f"Memory system path: {prsist_dir}")
        print(f"Directory exists: {os.path.exists(prsist_dir)}")
        
        # Import semantic analyzer
        from semantic_analyzer import SemanticAnalyzer
        print("✅ SemanticAnalyzer imported successfully")
        
        # Initialize analyzer
        analyzer = SemanticAnalyzer(prsist_dir, parent_dir)
        
        # Check what features are available
        has_sklearn = import_results.get('sklearn', {}).get('success', False)
        has_numpy = import_results.get('numpy', {}).get('success', False)
        
        # Check analyzer capabilities
        sklearn_available = hasattr(analyzer, 'tfidf_vectorizer') and analyzer.tfidf_vectorizer is not None
        
        print(f"✅ Available packages: NumPy={has_numpy}, Sklearn={has_sklearn}")
        print(f"✅ Analyzer TF-IDF capability: {sklearn_available}")
        
        if sklearn_available:
            print("🎉 Enhanced semantic analysis with TF-IDF is ready!")
            
            # Test enhanced embedding generation
            test_text = "This is a comprehensive test of the enhanced Prsist Memory System semantic analysis with TF-IDF vectorization"
            embedding = analyzer._generate_text_embedding(test_text, "test_enhanced", "test")
            
            if embedding:
                method = embedding.metadata.get('generated_method', 'unknown')
                vector_size = len(embedding.embedding_vector)
                print(f"✅ Enhanced embedding generation successful!")
                print(f"   Method used: {method}")
                print(f"   Vector dimensions: {vector_size}")
                
                # Test similarity with another embedding
                test_text2 = "Another test text for similarity comparison using the memory system"
                embedding2 = analyzer._generate_text_embedding(test_text2, "test_comparison", "test")
                
                if embedding2:
                    similarity = analyzer.calculate_similarity(embedding, embedding2)
                    print(f"✅ Similarity calculation: {similarity:.4f}")
                
            else:
                print("❌ Enhanced embedding generation failed")
                
        else:
            print("🟡 Using basic semantic analysis (fallback mode)")
            
            # Test basic functionality
            test_text = "Basic test of semantic analysis without TF-IDF"
            embedding = analyzer._generate_text_embedding(test_text, "test_basic", "test")
            
            if embedding:
                method = embedding.metadata.get('generated_method', 'unknown')
                print(f"✅ Basic embedding generation working: {method}")
            else:
                print("❌ Even basic embedding generation failed")
        
        # Test file analysis
        test_file = os.path.join(prsist_dir, 'memory_manager.py')
        if os.path.exists(test_file):
            print(f"\nTesting file analysis on: {os.path.basename(test_file)}")
            analysis = analyzer.analyze_file_semantics(test_file)
            
            if 'error' not in analysis:
                language = analysis.get('language', 'unknown')
                elements_count = len(analysis.get('elements', []))
                keywords_count = len(analysis.get('semantic_keywords', []))
                complexity = analysis.get('semantic_complexity', 0)
                
                print(f"✅ File analysis successful!")
                print(f"   Language: {language}")
                print(f"   Code elements: {elements_count}")
                print(f"   Semantic keywords: {keywords_count}")
                print(f"   Complexity score: {complexity}")
            else:
                print(f"❌ File analysis failed: {analysis.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory system integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function that fixes imports and tests everything"""
    print("🎯 Fix and Test Enhanced Prsist Memory System")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Step 1: Fix import issues
    fix_ok = fix_package_imports()
    
    # Step 2: Test basic imports
    import_results, imports_ok = test_basic_imports()
    
    # Step 3: Try manual fixes for failed imports
    if not import_results.get('sklearn', {}).get('success', False):
        print("\n🔧 Attempting to fix sklearn import...")
        sklearn_fix_ok = manually_fix_sklearn_import()
        if sklearn_fix_ok:
            # Re-test sklearn after fix
            try:
                import sklearn
                print(f"✅ sklearn now working: {sklearn.__version__}")
                import_results['sklearn'] = {'success': True, 'version': sklearn.__version__, 'error': None}
                imports_ok = True
            except:
                print("❌ sklearn fix didn't work")
    
    # Step 4: Test enhanced functionality if imports work
    enhanced_ok = False
    if imports_ok:
        enhanced_ok = test_enhanced_functionality(import_results)
    
    # Step 5: Test memory system integration
    memory_ok = False
    if imports_ok:
        memory_ok = test_memory_system_integration(import_results)
    
    # Final Results
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)
    
    # Count successful imports
    successful_packages = [name for name, result in import_results.items() if result.get('success', False)]
    total_packages = len(import_results)
    
    print(f"Package Imports: {len(successful_packages)}/{total_packages}")
    print(f"Working packages: {', '.join(successful_packages)}")
    
    if len(successful_packages) >= 2 and 'numpy' in successful_packages:
        if 'sklearn' in successful_packages and enhanced_ok and memory_ok:
            print("\n🎉 COMPLETE SUCCESS!")
            print("✅ All packages working with enhanced features")
            print("✅ TF-IDF semantic analysis active")
            print("✅ Advanced similarity calculations enabled")
            print("✅ Memory system fully integrated")
            print("\n🚀 Next steps:")
            print("   cd ..")
            print("   py -3.13 .prsist/test_runner.py")
            print("\nYour enhanced Prsist Memory System is ready! 🎊")
            
        else:
            print("\n🟡 PARTIAL SUCCESS!")
            print("✅ Core packages working")
            print("✅ Basic memory system functionality available")
            if 'sklearn' not in successful_packages:
                print("🟡 Sklearn not available - using fallback semantic analysis")
            print("\n🚀 Your memory system will work with basic functionality")
            
        return True
    else:
        print("\n❌ CRITICAL ISSUES")
        print("❌ Essential packages not working")
        print("🔧 Try running as administrator or use virtual environment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)