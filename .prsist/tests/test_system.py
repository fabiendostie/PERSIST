#!/usr/bin/env python3
"""
Test script for Prsist Memory System integration.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_memory_system():
    """Test the complete memory system integration."""
    print("Testing Prsist Memory System...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        import memory_manager
        import session_tracker
        import context_builder
        import database
        import utils
        
        MemoryManager = memory_manager.MemoryManager
        SessionTracker = session_tracker.SessionTracker
        ContextBuilder = context_builder.ContextBuilder
        MemoryDatabase = database.MemoryDatabase
        get_project_root = utils.get_project_root
        get_git_info = utils.get_git_info
        print("   + All modules imported successfully")
        
        # Test configuration loading
        print("2. Testing configuration...")
        memory_manager = MemoryManager()
        config = memory_manager.config
        print(f"   + Configuration loaded: {len(config)} sections")
        
        # Test system validation
        print("3. Testing system validation...")
        validation = memory_manager.validate_system()
        if validation["valid"]:
            print("   + System validation passed")
        else:
            print(f"   - System validation failed: {validation['issues']}")
            return False
        
        # Test database initialization
        print("4. Testing database...")
        db = MemoryDatabase()
        recent_sessions = db.get_recent_sessions(1)
        print(f"   + Database connected, found {len(recent_sessions)} recent sessions")
        
        # Test session creation
        print("5. Testing session creation...")
        session_result = memory_manager.start_session({"test": "integration_test"})
        if session_result.get("memory_system_active"):
            session_id = session_result["session_id"]
            print(f"   + Session created: {session_id[:8]}...")
        else:
            print(f"   - Session creation failed: {session_result.get('error')}")
            return False
        
        # Test context building
        print("6. Testing context building...")
        context = memory_manager.get_session_context()
        if context:
            print(f"   + Context built: {len(context)} characters")
        else:
            print("   - Context building failed")
            return False
        
        # Test tool usage logging
        print("7. Testing tool usage logging...")
        tool_logged = memory_manager.log_tool_usage(
            tool_name="Test",
            input_data={"test": "data"},
            output_data="test output",
            execution_time_ms=100,
            success=True
        )
        if tool_logged:
            print("   + Tool usage logged successfully")
        else:
            print("   - Tool usage logging failed")
            return False
        
        # Test file interaction logging
        print("8. Testing file interaction logging...")
        file_logged = memory_manager.log_file_interaction(
            file_path="test_file.py",
            action_type="edit",
            line_changes={"lines_added": 5, "lines_removed": 2}
        )
        if file_logged:
            print("   + File interaction logged successfully")
        else:
            print("   - File interaction logging failed")
            return False
        
        # Test checkpoint creation
        print("9. Testing checkpoint creation...")
        checkpoint_created = memory_manager.create_checkpoint("test_checkpoint")
        if checkpoint_created:
            print("   + Checkpoint created successfully")
        else:
            print("   - Checkpoint creation failed")
            return False
        
        # Test session info
        print("10. Testing session info...")
        session_info = memory_manager.get_session_info()
        if session_info.get("session_id"):
            print(f"   + Session info retrieved: {session_info['tool_usage_count']} tools used")
        else:
            print("   - Session info retrieval failed")
            return False
        
        # Test session ending
        print("11. Testing session ending...")
        session_ended = memory_manager.end_session(archive=True)
        if session_ended:
            print("   + Session ended and archived successfully")
        else:
            print("   - Session ending failed")
            return False
        
        print("\n+ All tests passed! Memory system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n- Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hooks():
    """Test the Claude Code hooks."""
    print("\nTesting Claude Code hooks...")
    
    try:
        # Test SessionStart hook
        print("1. Testing SessionStart hook...")
        hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "SessionStart.py"
        if hook_path.exists():
            print("   + SessionStart hook file exists")
            
            # Try to import and validate
            import subprocess
            result = subprocess.run([
                sys.executable, str(hook_path)
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    if "memory_system_active" in output:
                        print("   + SessionStart hook executed successfully")
                    else:
                        print("   - SessionStart hook output format invalid")
                        return False
                except json.JSONDecodeError:
                    print("   - SessionStart hook output not valid JSON")
                    return False
            else:
                print(f"   - SessionStart hook failed: {result.stderr}")
                return False
        else:
            print("   - SessionStart hook file not found")
            return False
        
        # Test PostToolUse hook
        print("2. Testing PostToolUse hook...")
        hook_path = Path(__file__).parent.parent / ".claude" / "hooks" / "PostToolUse.py"
        if hook_path.exists():
            print("   + PostToolUse hook file exists")
            
            # Test with sample data
            result = subprocess.run([
                sys.executable, str(hook_path),
                "Test", '{"test": "data"}', "test output", "0.1", "true"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("   + PostToolUse hook executed successfully")
            else:
                print(f"   - PostToolUse hook failed: {result.stderr}")
                return False
        else:
            print("   - PostToolUse hook file not found")
            return False
        
        print("\n+ Hook tests passed!")
        return True
        
    except Exception as e:
        print(f"\n- Hook test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_memory_system()
    if success:
        success = test_hooks()
    
    if success:
        print("\n[SUCCESS] Prsist Memory System Phase 1 implementation complete!")
        print("\nNext steps:")
        print("- Configure Claude Code to use the hooks")
        print("- Test with actual Claude Code sessions")
        print("- Monitor performance and adjust settings")
        print("- Plan Phase 2 enhancements")
    else:
        print("\n[FAILED] Some tests failed. Please check the errors above.")
        sys.exit(1)