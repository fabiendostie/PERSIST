#!/usr/bin/env python3
"""
Fix SciPy installation by replacing with correct wheel (no emojis)
"""

import sys
import os
import shutil
import zipfile
import site
from pathlib import Path

def remove_broken_scipy():
    """Remove the broken scipy installation"""
    print("Removing broken scipy installation...")
    
    user_site = site.getusersitepackages()
    user_site_path = Path(user_site)
    
    # Remove scipy directory
    scipy_dir = user_site_path / "scipy"
    if scipy_dir.exists():
        print(f"   Removing: {scipy_dir}")
        shutil.rmtree(scipy_dir)
    
    # Remove scipy.libs directory
    scipy_libs_dir = user_site_path / "scipy.libs"
    if scipy_libs_dir.exists():
        print(f"   Removing: {scipy_libs_dir}")
        shutil.rmtree(scipy_libs_dir)
    
    # Remove scipy dist-info
    scipy_dist_info = user_site_path / "scipy-1.16.1.dist-info"
    if scipy_dist_info.exists():
        print(f"   Removing: {scipy_dist_info}")
        shutil.rmtree(scipy_dist_info)
    
    print("Broken scipy removed")

def install_correct_scipy():
    """Install the correct scipy wheel"""
    print("Installing correct scipy wheel...")
    
    wheel_file = "scipy-1.16.1-cp313-cp313-win_amd64.whl"
    
    if not os.path.exists(wheel_file):
        print(f"ERROR: Wheel file {wheel_file} not found!")
        print("Please download it first with:")
        print('Invoke-WebRequest -Uri "https://files.pythonhosted.org/packages/21/12/c0efd2941f01940119b5305c375ae5c0fcb7ec193f806bd8f158b73a1782/scipy-1.16.1-cp313-cp313-win_amd64.whl" -OutFile "scipy-1.16.1-cp313-cp313-win_amd64.whl"')
        return False
    
    try:
        user_site = site.getusersitepackages()
        user_site_path = Path(user_site)
        
        with zipfile.ZipFile(wheel_file, 'r') as wheel:
            wheel.extractall(user_site_path)
            print(f"SUCCESS: Installed {wheel_file}")
            return True
            
    except Exception as e:
        print(f"ERROR: Failed to install {wheel_file}: {e}")
        return False

def test_scipy():
    """Test if scipy works now"""
    print("Testing scipy...")
    
    try:
        import scipy
        print(f"SUCCESS: SciPy imported successfully: {scipy.__version__}")
        
        # Test a basic function
        import scipy.spatial.distance
        result = scipy.spatial.distance.cosine([1, 2, 3], [4, 5, 6])
        print(f"SUCCESS: SciPy cosine distance test: {result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: SciPy test failed: {e}")
        return False

def test_sklearn():
    """Test if sklearn works now"""
    print("Testing sklearn...")
    
    try:
        import sklearn
        print(f"SUCCESS: Scikit-learn imported successfully: {sklearn.__version__}")
        
        # Test TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        texts = ["hello world", "goodbye world"]
        matrix = vectorizer.fit_transform(texts)
        print(f"SUCCESS: TF-IDF test successful: {matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Sklearn test failed: {e}")
        return False

def main():
    """Main function"""
    print("SciPy Fix Script")
    print("=" * 40)
    print(f"Python: {sys.version}")
    print(f"User site: {site.getusersitepackages()}")
    print()
    
    # Step 1: Remove broken scipy
    remove_broken_scipy()
    
    # Step 2: Install correct scipy
    scipy_ok = install_correct_scipy()
    
    if not scipy_ok:
        print("ERROR: Failed to install scipy")
        return False
    
    # Step 3: Test scipy
    scipy_test_ok = test_scipy()
    
    # Step 4: Test sklearn
    sklearn_test_ok = test_sklearn()
    
    # Results
    print("\n" + "=" * 40)
    print("Results:")
    print(f"   SciPy working: {'YES' if scipy_test_ok else 'NO'}")
    print(f"   Sklearn working: {'YES' if sklearn_test_ok else 'NO'}")
    
    if scipy_test_ok and sklearn_test_ok:
        print("\nSUCCESS! Both SciPy and Sklearn are now working!")
        print("Enhanced TF-IDF semantic analysis is now available")
        print("\nNext steps:")
        print("   py -3.13 fix_and_test.py  # Re-run the full test")
        print("   cd .. && py -3.13 .prsist/test_runner.py  # Run memory system tests")
        return True
    else:
        print("\nStill having issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)