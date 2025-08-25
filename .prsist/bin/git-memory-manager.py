#!/usr/bin/env python3
"""
Git Memory Manager - Handles memory system during git operations

This utility helps manage the active memory files during git operations by:
1. Creating snapshots of active memory state
2. Pausing memory system updates
3. Committing memory files with git operations
4. Resuming memory system after git operations
"""

import os
import sys
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
import subprocess

class GitMemoryManager:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.prsist_dir = self.project_root / ".prsist"
        self.memory_lock_file = self.prsist_dir / "storage" / ".memory_lock"
        self.snapshot_dir = self.prsist_dir / "temp" / "git_snapshots"
        
        # Ensure directories exist
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
    def is_memory_active(self):
        """Check if memory system is currently active"""
        session_file = self.prsist_dir / "sessions" / "active" / "current-session.json"
        return session_file.exists() and not self.memory_lock_file.exists()
    
    def create_memory_lock(self):
        """Create a lock file to pause memory system updates"""
        lock_data = {
            "locked_at": datetime.now().isoformat(),
            "locked_by": "git-memory-manager",
            "reason": "Git operation in progress"
        }
        
        with open(self.memory_lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        print("* Memory system paused for git operations")
    
    def remove_memory_lock(self):
        """Remove the lock file to resume memory system updates"""
        if self.memory_lock_file.exists():
            self.memory_lock_file.unlink()
            print("* Memory system resumed")
    
    def create_snapshot(self, snapshot_name=None):
        """Create a snapshot of current memory state"""
        if not snapshot_name:
            snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot_path = self.snapshot_dir / snapshot_name
        snapshot_path.mkdir(exist_ok=True)
        
        # Files to snapshot
        files_to_snapshot = [
            ".prsist/context/claude-context.md",
            ".prsist/sessions/active/current-session.json",
            ".prsist/storage/memory.log",
            ".prsist/storage/sessions.db",
            ".prsist/__pycache__/database.cpython-310.pyc",
            ".prsist/__pycache__/memory_manager.cpython-310.pyc"
        ]
        
        for file_path in files_to_snapshot:
            src = self.project_root / file_path
            if src.exists():
                dst = snapshot_path / Path(file_path).name
                shutil.copy2(src, dst)
        
        print(f"* Memory snapshot created: {snapshot_name}")
        return snapshot_path
    
    def restore_snapshot(self, snapshot_name):
        """Restore memory state from a snapshot"""
        snapshot_path = self.snapshot_dir / snapshot_name
        if not snapshot_path.exists():
            print(f"* Snapshot not found: {snapshot_name}")
            return False
        
        # Files to restore
        restore_mapping = {
            "claude-context.md": ".prsist/context/claude-context.md",
            "current-session.json": ".prsist/sessions/active/current-session.json",
            "memory.log": ".prsist/storage/memory.log",
            "sessions.db": ".prsist/storage/sessions.db",
            "database.cpython-310.pyc": ".prsist/__pycache__/database.cpython-310.pyc",
            "memory_manager.cpython-310.pyc": ".prsist/__pycache__/memory_manager.cpython-310.pyc"
        }
        
        for src_name, dst_path in restore_mapping.items():
            src = snapshot_path / src_name
            dst = self.project_root / dst_path
            if src.exists():
                shutil.copy2(src, dst)
        
        print(f"* Memory state restored from: {snapshot_name}")
        return True
    
    def commit_memory_state(self, commit_message="Update memory system state"):
        """Commit current memory state to git"""
        memory_files = [
            ".prsist/context/claude-context.md",
            ".prsist/sessions/active/current-session.json", 
            ".prsist/storage/memory.log",
            ".prsist/storage/sessions.db"
        ]
        
        # Add memory files to git (force add ignored files)
        for file_path in memory_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                subprocess.run(["git", "add", "--force", file_path], cwd=self.project_root)
        
        # Commit the changes
        subprocess.run([
            "git", "commit", "-m", 
            f"feat: {commit_message}\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        ], cwd=self.project_root)
        
        print("* Memory state committed to git")
    
    def safe_git_operation(self, git_command, commit_memory=True):
        """Perform git operation with memory system management"""
        try:
            # 1. Create snapshot
            snapshot_name = f"pre_{git_command[0]}_{datetime.now().strftime('%H%M%S')}"
            self.create_snapshot(snapshot_name)
            
            # 2. Pause memory system
            self.create_memory_lock()
            
            # 3. Commit current memory state if requested
            if commit_memory:
                self.commit_memory_state(f"memory state before {' '.join(git_command)}")
            
            # 4. Execute git command
            result = subprocess.run(git_command, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"* Git operation successful: {' '.join(git_command)}")
            else:
                print(f"* Git operation failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"* Error during git operation: {e}")
            return False
            
        finally:
            # Always resume memory system
            self.remove_memory_lock()

def main():
    if len(sys.argv) < 2:
        print("""
Git Memory Manager - Usage:

Commands:
  pause                    - Pause memory system
  resume                   - Resume memory system  
  snapshot [name]          - Create memory snapshot
  restore <name>           - Restore memory snapshot
  commit [message]         - Commit memory state
  safe-push [branch]       - Safe push with memory commit
  safe-merge <branch>      - Safe merge with memory commit
  status                   - Show memory system status
        """)
        return
    
    manager = GitMemoryManager()
    command = sys.argv[1]
    
    if command == "pause":
        manager.create_memory_lock()
        
    elif command == "resume":
        manager.remove_memory_lock()
        
    elif command == "snapshot":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        manager.create_snapshot(name)
        
    elif command == "restore":
        if len(sys.argv) < 3:
            print("Usage: git-memory-manager.py restore <snapshot_name>")
            return
        manager.restore_snapshot(sys.argv[2])
        
    elif command == "commit":
        message = sys.argv[2] if len(sys.argv) > 2 else "Update memory system state"
        manager.commit_memory_state(message)
        
    elif command == "safe-push":
        branch = sys.argv[2] if len(sys.argv) > 2 else "main"
        manager.safe_git_operation(["git", "push", "origin", branch])
        
    elif command == "safe-merge":
        if len(sys.argv) < 3:
            print("Usage: git-memory-manager.py safe-merge <branch>")
            return
        branch = sys.argv[2]
        manager.safe_git_operation(["git", "merge", branch])
        
    elif command == "status":
        active = manager.is_memory_active()
        lock_exists = manager.memory_lock_file.exists()
        print(f"Memory System Status:")
        print(f"  Active: {'Yes' if active else 'No'}")
        print(f"  Locked: {'Yes' if lock_exists else 'No'}")
        if lock_exists:
            with open(manager.memory_lock_file) as f:
                lock_data = json.load(f)
            print(f"  Locked at: {lock_data.get('locked_at')}")
            print(f"  Reason: {lock_data.get('reason')}")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()