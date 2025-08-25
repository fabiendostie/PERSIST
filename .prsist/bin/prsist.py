#!/usr/bin/env python3
"""
Prsist Memory System CLI - Simple, memorable commands
Usage: prsist [options]

Single Letter Options:
  -t  Test system
  -s  Status/session info  
  -c  Context (what Claude sees)
  -r  Recent sessions
  -h  Health check
  -f  Feature log (interactive)
  -m  Memory stats
  -v  Validate system
  -p  Project memory operations
  -d  Decisions (add decision)
  -e  End session
  -n  New session (start)
  -k  Checkpoint (create)
  -x  Export session data
  -z  Cleanup old data
  -l  List all commands
  -a  All checks (equivalent to -tschrmv)
  -i  Force context injection (failsafe)
  
Chain commands: prsist -tsc (test + status + context)
Examples:
  prsist -t        Test system
  prsist -h        Health check
  prsist -tsc      Test + Status + Context
  prsist -a        Run all checks
  prsist -hm       Health + Memory stats
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
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
        print(f"[ERROR] Test error: {e}")
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
        print(f"[ERROR] Status error: {e}")
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
        print(f"[ERROR] Context error: {e}")
        return False

def force_context_injection():
    """Force inject project context (failsafe)"""
    print("[FORCE-CONTEXT] Force injecting project context...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        
        # Force rebuild context
        context = mm.get_session_context()
        print(f"  * Context loaded: {len(context)} characters")
        
        # Display current project info
        print("  * Project Memory:")
        context_file = Path(".prsist/context/claude-context.md")
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")
                        if 'Memory System Status' in line:
                            break
        
        # Display session info
        session_file = Path(".prsist/sessions/active/current-session.json")
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                print(f"  * Session: {session_data.get('session_id', 'Unknown')[:8]}")
                print(f"  * Tools used: {session_data.get('tool_count', 0)}")
        
        print("  * Context injection complete - memory should now be available")
        return True
        
    except Exception as e:
        print(f"  * Context injection failed: {e}")
        print("  Manual fallback: Read .prsist/context/claude-context.md")
        return False

def recent_sessions():
    """Show recent sessions with enhanced descriptions"""
    print("[RECENT] Recent Sessions...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        sessions = mm.get_recent_sessions(5)
        
        if sessions:
            # Check for current active session
            current_session_id = None
            try:
                current_context = mm.get_session_context()
                current_session_id = current_context.get("session_id")
            except:
                pass
            
            for i, session in enumerate(sessions):
                # Generate meaningful description
                description = generate_session_description(session)
                session_id = session.get("session_id", "Unknown")
                short_id = session_id[:8] if session_id != "Unknown" else "Unknown"
                
                if i == 0 and session_id == current_session_id:
                    # Current active session - enhanced display
                    print(f"🔄 Active Session ({short_id})")
                    print(f"  - Activity: {description}")
                    print(f"  - Focus: Enhanced session descriptions and memory system improvements")
                    print(f"  - Status: Currently active")
                    print("")
                    print("📖 Previous Sessions:")
                elif i == 0:
                    # First session but not current
                    print(f"  - {short_id}: {description}")
                else:
                    # Previous sessions
                    print(f"  - {short_id}: {description}")
        else:
            print("  No recent sessions found")
        return True
    except Exception as e:
        print(f"[ERROR] Recent sessions error: {e}")
        return False

def generate_session_description(session_data):
    """Generate a meaningful description for a session based on its activity"""
    try:
        # Get session details for rich context generation
        session_id = session_data.get('session_id')
        if not session_id:
            return "Unknown session"
            
        # Try to load full session data from archived or active sessions
        memory_dir = Path(__file__).parent.parent
        archived_file = memory_dir / "sessions" / "archived" / f"{session_id}.json"
        
        full_session_data = None
        if archived_file.exists():
            with open(archived_file, 'r', encoding='utf-8') as f:
                full_session_data = json.load(f)
        else:
            # Try active session file
            active_file = memory_dir / "sessions" / "active" / "current-session.json"
            if active_file.exists():
                with open(active_file, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
                    if temp_data.get('id') == session_id:
                        full_session_data = temp_data
        
        if full_session_data:
            # Extract rich context from session data
            description = _extract_contextual_description(full_session_data, session_data)
            if description:
                return description
        
        # Fallback to enhanced generic descriptions
        tool_count = session_data.get('tool_usage_count', 0)
        file_count = session_data.get('file_interaction_count', 0)
        tools_used = session_data.get('unique_tools_used', [])
        
        # Generate enhanced descriptions
        if file_count > 0 and tool_count > 0:
            if 'Write' in tools_used and 'Edit' in tools_used:
                return f"Created and refined {file_count} file{'s' if file_count > 1 else ''} ({tool_count} operations)"
            elif 'Write' in tools_used:
                return f"New file development - {file_count} file{'s' if file_count > 1 else ''} created"
            elif 'Edit' in tools_used or 'MultiEdit' in tools_used:
                return f"Code editing session - {file_count} file{'s' if file_count > 1 else ''} modified"
            elif 'Read' in tools_used:
                return f"Code analysis and exploration ({tool_count} operations)"
        elif tool_count > 15:
            if 'Read' in tools_used and 'Grep' in tools_used:
                return f"Extensive codebase exploration ({tool_count} operations)"
            elif 'Bash' in tools_used:
                return f"Command-heavy development session ({tool_count} operations)"
        elif tool_count > 5:
            if 'Bash' in tools_used:
                return f"Command execution and testing ({tool_count} operations)"
            elif 'Read' in tools_used:
                return f"Documentation and code review ({tool_count} operations)"
        elif tool_count > 0:
            return f"Light development work ({tool_count} operations)"
        
        return "Empty session"
    except Exception as e:
        logging.debug(f"Failed to generate session description: {e}")
        return "Development session"


def _extract_contextual_description(full_session_data, session_summary):
    """Extract meaningful context from full session data"""
    try:
        tool_usage = full_session_data.get('tool_usage', [])
        file_interactions = full_session_data.get('file_interactions', [])
        
        # Extract key information
        files_modified = []
        files_read = []
        commands_run = []
        searches_made = []
        todos_worked = []
        
        # Analyze tool usage for context
        for tool in tool_usage:
            tool_name = tool.get('tool_name', '')
            input_data = tool.get('input_data', {})
            
            if tool_name in ['Edit', 'MultiEdit', 'Write']:
                file_path = input_data.get('file_path', '')
                if file_path:
                    files_modified.append(Path(file_path).name)
            elif tool_name == 'Read':
                file_path = input_data.get('file_path', '')
                if file_path:
                    files_read.append(Path(file_path).name)
            elif tool_name == 'Bash':
                command = input_data.get('command', '')
                if command:
                    commands_run.append(command)
            elif tool_name == 'WebSearch':
                query = input_data.get('query', '')
                if query:
                    searches_made.append(query)
            elif tool_name == 'TodoWrite':
                todos = input_data.get('todos', [])
                for todo in todos:
                    content = todo.get('content', '')
                    if content and content not in todos_worked:
                        todos_worked.append(content)
        
        # Generate contextual description based on extracted data
        description_parts = []
        
        # Main activity identification
        if searches_made:
            for query in searches_made[:1]:  # Focus on first search
                if 'mcp' in query.lower() or 'context7' in query.lower():
                    description_parts.append("Researched and integrated Context7 MCP server")
                    break
                elif 'memory' in query.lower():
                    description_parts.append("Researched memory system implementation")
                    break
                else:
                    description_parts.append(f"Researched {query[:50]}")
                    break
        
        if todos_worked:
            todo_descriptions = []
            for todo in todos_worked[:2]:  # Show first 2 todos
                if 'mcp' in todo.lower() or 'context7' in todo.lower():
                    todo_descriptions.append("MCP server setup")
                elif 'enhance' in todo.lower() or 'improve' in todo.lower():
                    todo_descriptions.append("system improvements")
                else:
                    todo_descriptions.append(todo[:30])
            
            if not description_parts:  # Only if no search activity found
                description_parts.append(f"Worked on: {', '.join(todo_descriptions)}")
        
        # File modification context
        if files_modified:
            unique_files = list(dict.fromkeys(files_modified))  # Remove duplicates
            if len(unique_files) == 1:
                file_name = unique_files[0]
                if 'activity_analyzer' in file_name:
                    description_parts.append(f"Enhanced session analysis in {file_name}")
                elif 'prsist' in file_name:
                    description_parts.append(f"Improved memory CLI in {file_name}")
                elif 'memory' in file_name or 'session' in file_name:
                    description_parts.append(f"Updated memory system in {file_name}")
                else:
                    description_parts.append(f"Modified {file_name}")
            elif len(unique_files) <= 3:
                description_parts.append(f"Modified {', '.join(unique_files[:2])} + {len(unique_files)-2} more files" if len(unique_files) > 2 else f"Modified {', '.join(unique_files)}")
            else:
                description_parts.append(f"Modified {len(unique_files)} files including {', '.join(unique_files[:2])}")
        
        # Command execution context
        if commands_run and not description_parts:
            command_contexts = []
            for cmd in commands_run[:3]:  # Look at first 3 commands
                if 'mcp add' in cmd:
                    command_contexts.append("MCP server installation")
                elif 'mem' in cmd and 'recent' in cmd:
                    command_contexts.append("memory system testing")
                elif 'python' in cmd and 'prsist' in cmd:
                    command_contexts.append("memory CLI validation")
                elif 'npm' in cmd or 'npx' in cmd:
                    command_contexts.append("package management")
            
            if command_contexts:
                description_parts.append(', '.join(dict.fromkeys(command_contexts)))  # Remove duplicates
        
        # File reading context (only if no modifications)
        if files_read and not files_modified and not description_parts:
            unique_read = list(dict.fromkeys(files_read))
            if len(unique_read) == 1:
                description_parts.append(f"Analyzed {unique_read[0]}")
            elif len(unique_read) <= 3:
                description_parts.append(f"Reviewed {', '.join(unique_read)}")
            else:
                description_parts.append(f"Explored {len(unique_read)} files")
        
        # Combine description parts
        if description_parts:
            main_description = description_parts[0]
            
            # Add file context if not already included
            if files_modified and 'Modified' not in main_description and 'Enhanced' not in main_description:
                if len(files_modified) == 1:
                    main_description += f" - modified {files_modified[0]}"
                elif len(files_modified) <= 3:
                    main_description += f" - modified {len(files_modified)} files"
            
            return main_description
        
        return None
    except Exception as e:
        logging.debug(f"Failed to extract contextual description: {e}")
        return None

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
        print(f"[ERROR] Health check error: {e}")
        return False

def feature_log():
    """Interactive feature logging"""
    print("[FEATURE] Feature Logging...")
    try:
        feature_name = input("Feature name: ").strip()
        if not feature_name:
            print("[ERROR] Feature name required")
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
        print("\n[CANCEL] Feature logging cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] Feature logging error: {e}")
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
        print(f"[ERROR] Memory stats error: {e}")
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
        print(f"[ERROR] Validation error: {e}")
        return False

def project_memory():
    """Add to project memory"""
    print("[PROJECT] Project Memory...")
    try:
        content = input("Enter information to add to project memory: ").strip()
        if not content:
            print("[ERROR] Content required")
            return False
            
        from memory_manager import MemoryManager
        mm = MemoryManager()
        success = mm.add_project_memory(content)
        
        if success:
            print(f"[PASS] Project memory updated")
            return True
        else:
            print("[FAIL] Failed to update project memory")
            return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Project memory update cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] Project memory error: {e}")
        return False

def add_decision():
    """Add a decision record"""
    print("[DECISION] Add Decision...")
    try:
        title = input("Decision title: ").strip()
        if not title:
            print("[ERROR] Decision title required")
            return False
            
        description = input("Description: ").strip()
        if not description:
            print("[ERROR] Description required")
            return False
            
        category = input("Category (architecture/technical/process/design): ").strip()
        if not category:
            category = "general"
            
        impact = input("Impact (low/medium/high/critical): ").strip()
        if not impact:
            impact = "medium"
            
        from memory_manager import MemoryManager
        mm = MemoryManager()
        success = mm.add_decision(title, description, category, impact)
        
        if success:
            print(f"[PASS] Decision '{title}' recorded")
            return True
        else:
            print("[FAIL] Failed to record decision")
            return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Decision recording cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] Decision recording error: {e}")
        return False

def end_session():
    """End current session"""
    print("[END] Ending Session...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        result = mm.end_session(archive=True)
        
        if result:
            print("[PASS] Session ended and archived")
            return True
        else:
            print("[WARN] No active session to end")
            return True
    except Exception as e:
        print(f"[ERROR] End session error: {e}")
        return False

def new_session():
    """Start new session"""
    print("[NEW] Starting New Session...")
    try:
        context = input("Session context (optional): ").strip()
        
        from memory_manager import MemoryManager
        mm = MemoryManager()
        
        session_data = {}
        if context:
            session_data["context"] = context
            
        result = mm.start_session(session_data)
        
        if result.get("memory_system_active"):
            session_id = result.get("session_id", "unknown")
            print(f"[PASS] New session started: {session_id[:8]}...")
            return True
        else:
            print(f"[FAIL] Failed to start session: {result.get('error', 'Unknown error')}")
            return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Session start cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] New session error: {e}")
        return False

def create_checkpoint():
    """Create a checkpoint"""
    print("[CHECKPOINT] Creating Checkpoint...")
    try:
        name = input("Checkpoint name (optional): ").strip()
        if not name:
            name = None
            
        from memory_manager import MemoryManager
        mm = MemoryManager()
        result = mm.create_checkpoint(name)
        
        if result:
            print(f"[PASS] Checkpoint created: {name or 'auto'}")
            return True
        else:
            print("[FAIL] Failed to create checkpoint")
            return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Checkpoint creation cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] Checkpoint error: {e}")
        return False

def export_session():
    """Export current session data"""
    print("[EXPORT] Exporting Session Data...")
    try:
        from memory_manager import MemoryManager
        mm = MemoryManager()
        data = mm.export_session_data(format="json")
        
        if data:
            filename = f"session_export_{mm.get_session_info().get('session_id', 'unknown')[:8]}.json"
            with open(filename, 'w') as f:
                f.write(data)
            print(f"[PASS] Session exported to {filename}")
            return True
        else:
            print("[WARN] No active session to export")
            return True
    except Exception as e:
        print(f"[ERROR] Export error: {e}")
        return False

def cleanup_data():
    """Clean up old data"""
    print("[CLEANUP] Cleaning Up Old Data...")
    try:
        days = input("Retention days (default 30): ").strip()
        if not days:
            days = 30
        else:
            days = int(days)
            
        from memory_manager import MemoryManager
        mm = MemoryManager()
        result = mm.cleanup_old_data(retention_days=days)
        
        print(f"[PASS] Cleanup completed: {result}")
        return True
    except ValueError:
        print("[ERROR] Invalid number of days")
        return False
    except KeyboardInterrupt:
        print("\n[CANCEL] Cleanup cancelled")
        return False
    except Exception as e:
        print(f"[ERROR] Cleanup error: {e}")
        return False

def list_commands():
    """List all available commands"""
    print("[HELP] Prsist Memory System Commands:")
    print("\n  Core Operations:")
    print("  -t  Test system")
    print("  -s  Status/session info")
    print("  -c  Context (what Claude sees)")
    print("  -r  Recent sessions")
    print("  -h  Health check")
    print("  -m  Memory stats")
    print("  -v  Validate system")
    print("\n  Session Management:")
    print("  -n  New session (start)")
    print("  -e  End session")
    print("  -k  Checkpoint (create)")
    print("  -x  Export session data")
    print("\n  Data Management:")
    print("  -f  Feature log (interactive)")
    print("  -p  Project memory (add)")
    print("  -d  Decisions (add decision)")
    print("  -z  Cleanup old data")
    print("\n  Shortcuts:")
    print("  -a  All core checks (equivalent to -tschrmv)")
    print("  -l  List commands (this help)")
    print("\nExamples:")
    print("  prsist -t      (test system)")
    print("  prsist -hm     (health + memory stats)")
    print("  prsist -tsc    (test + status + context)")
    print("  prsist -a      (run all core checks)")
    print("  prsist -nf     (new session + feature log)")
    return True

# Command mapping
COMMAND_MAP = {
    't': test_system,
    's': session_status,
    'c': show_context,
    'r': recent_sessions,
    'h': health_check,
    'f': feature_log,
    'm': memory_stats,
    'v': validate_system,
    'p': project_memory,
    'd': add_decision,
    'e': end_session,
    'n': new_session,
    'k': create_checkpoint,
    'x': export_session,
    'z': cleanup_data,
    'l': list_commands,
    'i': force_context_injection
}

def main():
    # Custom argument parser that handles combined flags like -tsc
    if len(sys.argv) < 2:
        print("Prsist Memory System CLI")
        print("Usage: prsist [options]")
        print("Example: prsist -t       (test)")
        print("Example: prsist -tsc     (test + status + context)")
        print("Example: prsist -a       (all checks)")
        print("Use 'prsist -l' to list all commands")
        return

    # Parse arguments manually to handle combined flags
    args = sys.argv[1]
    if args.startswith('-'):
        commands = args[1:]  # Remove the dash
    else:
        commands = args

    # Handle special 'all' command
    if 'a' in commands:
        commands = commands.replace('a', 'tschrmv')

    # Remove duplicates while preserving order
    seen = set()
    unique_commands = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            unique_commands.append(cmd)
    
    commands = ''.join(unique_commands)
    
    success_count = 0
    total_count = 0
    
    print(f"[RUN] Prsist Memory: Running {len(commands)} command(s)")
    print("-" * 50)
    
    for i, cmd in enumerate(commands):
        if cmd in COMMAND_MAP:
            if i > 0:
                print()  # Space between commands
            success = COMMAND_MAP[cmd]()
            total_count += 1
            if success:
                success_count += 1
        else:
            print(f"[ERROR] Unknown command: -{cmd}")
            total_count += 1
    
    print("-" * 50)
    print(f"[DONE] {success_count}/{total_count} commands completed successfully")

if __name__ == "__main__":
    main()