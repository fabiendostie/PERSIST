#!/usr/bin/env python3
"""
Setup Auto-Start for Prsist Memory System with Claude Code
Creates the necessary configuration for transparent integration
ASCII-safe version for Windows compatibility
"""

import os
import sys
import json
from pathlib import Path
import subprocess

class PrsistAutoStartSetup:
    def __init__(self):
        self.prsist_root = Path(__file__).parent
        self.project_root = Path.cwd()
        
    def create_claude_md_integration(self):
        """Add Prsist integration to CLAUDE.md file"""
        claude_md_path = self.project_root / "CLAUDE.md"
        
        prsist_section = """
## Prsist Memory System Integration

The Prsist Memory System is automatically active for Claude Code sessions. It provides:

- **Project Memory**: Persistent memory across conversations
- **Context Tracking**: Automatic context updates as you work  
- **Decision Logging**: Track important project decisions
- **Session Management**: Correlate work across sessions

### Commands for Claude

When needed, Claude can use these memory commands:

```bash
# Check memory status
python .prsist/bin/prsist.py -h

# View current context
python .prsist/bin/prsist.py -c

# Add project memory
python .prsist/bin/prsist.py -p

# Create checkpoint
python .prsist/bin/prsist.py -k
```

### Transparent Operation

The system runs transparently in the background:
- Auto-starts with Claude Code sessions
- Updates context after tool usage
- Maintains session history
- No user interaction required
"""
        
        try:
            if claude_md_path.exists():
                with open(claude_md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "Prsist Memory System" not in content:
                    # Add the section at the end
                    with open(claude_md_path, 'a', encoding='utf-8') as f:
                        f.write(prsist_section)
                    print("[SUCCESS] Added Prsist integration to CLAUDE.md")
                else:
                    print("[INFO] Prsist integration already exists in CLAUDE.md")
            else:
                print("[WARNING] CLAUDE.md not found - integration information not added")
                
        except Exception as e:
            print(f"[ERROR] Failed to update CLAUDE.md: {e}")
    
    def setup_environment_variables(self):
        """Setup environment variables for the current session"""
        try:
            os.environ['PRSIST_ACTIVE'] = 'true'
            os.environ['PRSIST_ROOT'] = str(self.prsist_root)
            os.environ['PRSIST_CONTEXT_FILE'] = str(self.prsist_root / 'context' / 'claude-context.md')
            os.environ['PRSIST_AUTO_START'] = 'true'
            
            print("[SUCCESS] Environment variables configured")
            print(f"   PRSIST_ACTIVE=true")
            print(f"   PRSIST_ROOT={self.prsist_root}")
            
        except Exception as e:
            print(f"[ERROR] Failed to set environment variables: {e}")
    
    def create_startup_trigger(self):
        """Create a startup trigger that can be called by Claude Code"""
        try:
            # Run the integration script to test it works
            integration_script = self.prsist_root / 'bin' / 'claude-integration.py'
            result = subprocess.run([
                sys.executable, str(integration_script)
            ], capture_output=True, text=True)
            
            print("Testing Prsist integration:")
            print(result.stdout)
            
            if result.returncode == 0:
                print("[SUCCESS] Integration test successful")
            else:
                print("[WARNING] Integration test completed with warnings")
                
        except Exception as e:
            print(f"[ERROR] Integration test failed: {e}")
    
    def verify_installation(self):
        """Verify the Prsist system is properly installed"""
        checks = []
        
        # Check core files exist
        required_files = [
            'bin/prsist.py',
            'hooks/PostToolUse.py', 
            'hooks/SessionStart.py',
            'hooks/claude-context-injector.py',
            'config/session-start.json'
        ]
        
        for file_path in required_files:
            full_path = self.prsist_root / file_path
            if full_path.exists():
                checks.append(f"[OK] {file_path}")
            else:
                checks.append(f"[MISSING] {file_path}")
        
        # Check if we can import memory system
        try:
            sys.path.insert(0, str(self.prsist_root))
            from memory_manager import MemoryManager
            checks.append("[OK] Memory manager importable")
            
            # Test basic functionality
            memory = MemoryManager()
            checks.append("[OK] Memory manager initializable")
            
        except Exception as e:
            checks.append(f"[ERROR] Memory manager not working: {e}")
        
        print("\nInstallation Verification:")
        for check in checks:
            print(f"   {check}")
        
        # Overall status
        failed_checks = [c for c in checks if "[MISSING]" in c or "[ERROR]" in c]
        if not failed_checks:
            print("\n[SUCCESS] Prsist Memory System is ready for Claude Code!")
            return True
        else:
            print(f"\n[WARNING] {len(failed_checks)} issues found - system may not work correctly")
            return False
    
    def run_setup(self):
        """Run the complete auto-start setup"""
        print("Setting up Prsist Memory System auto-start for Claude Code")
        print("=" * 60)
        
        # Verify installation
        if not self.verify_installation():
            print("\n[ERROR] Setup aborted due to installation issues")
            return False
        
        print("\nConfiguring integration...")
        
        # Setup components
        self.create_claude_md_integration()
        self.setup_environment_variables()
        self.create_startup_trigger()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("\nPrsist Memory System will now:")
        print("  * Auto-start with Claude Code sessions")
        print("  * Provide context automatically") 
        print("  * Track your development work")
        print("  * Update memory after tool usage")
        print("\nThe integration runs transparently in the background.")
        
        return True

def main():
    setup = PrsistAutoStartSetup()
    success = setup.run_setup()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())