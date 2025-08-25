#!/usr/bin/env python3
"""
Direct wheel installer for Python 3.13
Bypasses pip to install packages directly
"""

import zipfile
import sys
import os
from pathlib import Path

def install_wheel(wheel_path):
    """Install a wheel file directly to user site-packages"""
    print(f"Installing {wheel_path}...")
    try:
        with zipfile.ZipFile(wheel_path, 'r') as wheel:
            # Use user site-packages instead of system site-packages
            import site
            user_site = site.getusersitepackages()
            user_site_path = Path(user_site)
            user_site_path.mkdir(parents=True, exist_ok=True)
            
            print(f"   Installing to: {user_site_path}")
            wheel.extractall(user_site_path)
            print(f'✅ Installed {wheel_path}')
            return True
    except Exception as e:
        print(f'❌ Failed to install {wheel_path}: {e}')
        return False

def main():
    """Main installation function"""
    print("🚀 Installing Python packages to user site-packages")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Show where packages will be installed
    import site
    user_site = site.getusersitepackages()
    print(f"Install location: {user_site}")
    
    # Install in dependency order
    wheels = [
        'pip-25.2-py3-none-any.whl',
        'numpy-2.3.2-cp313-cp313-win_amd64.whl', 
        'joblib-1.5.1-py3-none-any.whl',
        'threadpoolctl-3.6.0-py3-none-any.whl',
        'scipy-1.16.1-cp313-cp313t-win_amd64.whl',
        'scikit_learn-1.7.1-cp313-cp313-win_amd64.whl'
    ]
    
    installed = 0
    failed = 0
    
    for wheel in wheels:
        if os.path.exists(wheel):
            if install_wheel(wheel):
                installed += 1
            else:
                failed += 1
        else:
            print(f'❌ Wheel {wheel} not found - skipping')
            failed += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   ✅ Installed: {installed}")
    print(f"   ❌ Failed/Missing: {failed}")
    
    if installed > 0:
        print(f"\n🎉 Successfully installed {installed} packages!")
        print("Now testing imports...")
        
        # Test imports
        test_imports = [
            ('pip', 'pip'),
            ('numpy', 'numpy'),
            ('scipy', 'scipy'), 
            ('sklearn', 'scikit-learn'),
            ('joblib', 'joblib'),
            ('threadpoolctl', 'threadpoolctl')
        ]
        
        for module, name in test_imports:
            try:
                __import__(module)
                print(f"   ✅ {name} imports successfully")
            except ImportError as e:
                print(f"   ❌ {name} import failed: {e}")

if __name__ == "__main__":
    main()