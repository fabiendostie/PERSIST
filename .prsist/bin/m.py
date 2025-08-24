#!/usr/bin/env python3
"""
Memory System CLI - Simple, memorable commands
Usage: python m.py [commands]

Single Letter Commands:
  t = Test system
  s = Status/session info  
  c = Context (what Claude sees)
  r = Recent sessions
  h = Health check
  f = Feature log (interactive)
  m = Memory stats
  v = Validate system
  l = List all commands
  
Chain commands: python m.py tsc (test + status + context)
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_system():
    """Test complete memory system"""
    print("[TEST] Testing Memory System...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "../tests/test_system.py"], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print("[PASS] All tests passed!")
            return True
        else:
            print(f"[FAIL] Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] Test error: {e}")
        return False

def session_status():
    """Get current session status"""
    print("[STATUS] Session Status...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        info = mm.get_session_info()
        
        if info.get("session_id"):
            print(f"  Active Session: {info['session_id'][:8]}...")
            print(f"  Tools Used: {info.get('tool_usage_count', 0)}")
            print(f"  Files Modified: {info.get('file_interaction_count', 0)}")
            print(f"  Duration: {info.get('duration_minutes', 0)} minutes")
        else:
            print("  No active session")
        return True
    except Exception as e:
        print(f"[FAIL] Status error: {e}")
        return False

def show_context():
    """Show current context (what Claude sees)"""
    print("[CONTEXT] Current Context...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        context = mm.get_session_context()
        
        if context:
            lines = context.split('\n')
            print(f"  Context Length: {len(context)} characters")
            print(f"  Lines: {len(lines)}")
            print("  Preview:")
            for i, line in enumerate(lines[:5]):
                print(f"    {line[:70]}{'...' if len(line) > 70 else ''}")
            if len(lines) > 5:
                print(f"    ... and {len(lines) - 5} more lines")
        else:
            print("  No context available")
        return True
    except Exception as e:
        print(f"[FAIL] Context error: {e}")
        return False

def recent_sessions():
    """Show recent sessions"""
    print("[RECENT] Recent Sessions...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        sessions = mm.get_recent_sessions(5)
        
        if sessions:
            for session in sessions:
                start_time = session.get("start_time", "Unknown")
                session_id = session.get("id", session.get("session_id", "Unknown"))
                print(f"  {start_time} - {session_id[:8] if session_id != 'Unknown' else 'Unknown'}")
        else:
            print("  No recent sessions found")
        return True
    except Exception as e:
        print(f"[FAIL] Recent sessions error: {e}")
        return False

def health_check():
    """System health check"""
    print("[HEALTH] Health Check...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        validation = mm.validate_system()
        
        if validation["valid"]:
            print("[PASS] System healthy")
        else:
            print("[WARN] System issues found:")
            for issue in validation.get("issues", []):
                print(f"    - {issue}")
        return validation["valid"]
    except Exception as e:
        print(f"[FAIL] Health check error: {e}")
        return False

def feature_log():
    """Interactive feature logging"""
    print("[FEATURE] Feature Logging...")
    try:
        feature_name = input("Feature name: ").strip()
        if not feature_name:
            print("[FAIL] Feature name required")
            return False
            
        description = input("Description (optional): ").strip()
        
        import subprocess
        cmd = [sys.executable, "hooks/FeatureComplete.py", feature_name]
        if description:
            cmd.append(description)
            
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"[PASS] Feature '{feature_name}' logged successfully")
            return True
        else:
            print(f"[FAIL] Feature logging failed: {result.stderr}")
            return False
    except KeyboardInterrupt:
        print("\n[FAIL] Feature logging cancelled")
        return False
    except Exception as e:
        print(f"[FAIL] Feature logging error: {e}")
        return False

def memory_stats():
    """Show memory system statistics"""
    print("[STATS] Memory Stats...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        stats = mm.get_memory_stats()
        
        print(f"  Total Sessions: {stats.get('total_sessions', 0)}")
        print(f"  Database Size: {stats.get('database_size_mb', 0)} MB")
        print(f"  Active Session: {stats.get('active_session', 'None')}")
        print(f"  Project Root: {stats.get('project_root', 'Unknown')}")
        return True
    except Exception as e:
        print(f"[FAIL] Memory stats error: {e}")
        return False

def validate_system():
    """Validate system integrity"""
    print("[VALIDATE] System Validation...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        validation = mm.validate_system()
        
        print(f"  System Valid: {validation['valid']}")
        if not validation["valid"]:
            print("  Issues:")
            for issue in validation.get("issues", []):
                print(f"    - {issue}")
        return validation["valid"]
    except Exception as e:
        print(f"[FAIL] Validation error: {e}")
        return False

def list_commands():
    """List all available commands"""
    print("[HELP] Available Commands:")
    print("  t = Test system")
    print("  s = Status/session info")
    print("  c = Context (what Claude sees)")
    print("  r = Recent sessions")
    print("  h = Health check")
    print("  f = Feature log (interactive)")
    print("  m = Memory stats")
    print("  v = Validate system")
    print("  l = List commands")
    print("\nChain commands: python m.py tsc")
    print("Example: python m.py t   (test system)")
    print("Example: python m.py hm  (health + memory stats)")
    return True

# Command mapping
COMMANDS = {
    't': test_system,
    's': session_status,
    'c': show_context,
    'r': recent_sessions,
    'h': health_check,
    'f': feature_log,
    'm': memory_stats,
    'v': validate_system,
    'l': list_commands
}

def main():
    if len(sys.argv) < 2:
        print("Memory System CLI")
        print("Usage: python m.py [commands]")
        print("Example: python m.py t     (test)")
        print("Example: python m.py tsc   (test + status + context)")
        print("Use 'python m.py l' to list all commands")
        return
    
    commands = sys.argv[1].lower()
    success_count = 0
    total_count = 0
    
    print(f"[RUN] Running {len(commands)} command(s): {' '.join(commands)}")
    print("-" * 50)
    
    for i, cmd in enumerate(commands):
        if cmd in COMMANDS:
            if i > 0:
                print()  # Space between commands
            success = COMMANDS[cmd]()
            total_count += 1
            if success:
                success_count += 1
        else:
            print(f"[FAIL] Unknown command: {cmd}")
            total_count += 1
    
    print("-" * 50)
    print(f"[PASS] {success_count}/{total_count} commands completed successfully")

if __name__ == "__main__":
    main()