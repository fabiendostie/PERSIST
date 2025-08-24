#!/usr/bin/env python3
"""
Git memory sync hook for Prsist Memory System.
Synchronizes memory context before push.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add memory system to Python path
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

try:
    from git_integration import GitMetadataExtractor
    from database import MemoryDatabase
    from session_tracker import SessionTracker
    from utils import setup_logging, get_project_root
except ImportError as e:
    print(f"Memory system not available: {e}", file=sys.stderr)
    sys.exit(0)

def sync_memory_before_push(branch_name: str) -> dict:
    """Synchronize memory context before push."""
    try:
        setup_logging("WARNING")  # Quiet for hooks
        
        project_root = get_project_root()
        
        # Initialize components
        git_extractor = GitMetadataExtractor(str(project_root))
        memory_db = MemoryDatabase(str(memory_dir / "storage" / "sessions.db"))
        session_tracker = SessionTracker(str(memory_dir))
        
        # Get current session and branch info
        current_session = session_tracker.get_current_session()
        current_branch = git_extractor.get_current_branch()
        
        sync_results = {
            "session_checkpoint": False,
            "branch_context_updated": False,
            "memory_validated": False,
            "sync_metadata_created": False
        }
        
        # Create pre-push checkpoint
        if current_session:
            checkpoint_name = f"pre_push_{branch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_success = session_tracker.create_checkpoint(checkpoint_name)
            sync_results["session_checkpoint"] = checkpoint_success
        
        # Update branch context with latest session data
        if current_session:
            branch_update_success = update_branch_context_for_push(
                memory_db, current_session, current_branch
            )
            sync_results["branch_context_updated"] = branch_update_success
        
        # Validate memory consistency
        validation_result = validate_memory_consistency(memory_db, git_extractor)
        sync_results["memory_validated"] = validation_result["valid"]
        
        # Create sync metadata
        sync_metadata = create_sync_metadata(git_extractor, current_session, branch_name)
        metadata_success = store_sync_metadata(memory_db, sync_metadata)
        sync_results["sync_metadata_created"] = metadata_success
        
        result = {
            "success": True,
            "branch": branch_name,
            "current_branch": current_branch,
            "sync_results": sync_results,
            "sync_metadata": sync_metadata,
            "validation": validation_result
        }
        
        logging.info(f"Memory sync completed for push to {branch_name}")
        return result
        
    except Exception as e:
        logging.error(f"Failed to sync memory before push: {e}")
        return {"success": False, "error": str(e)}

def update_branch_context_for_push(memory_db: MemoryDatabase, 
                                  current_session: dict, 
                                  branch_name: str) -> bool:
    """Update branch context with latest session data."""
    try:
        # Get current branch context
        branch_context = memory_db.get_branch_context(branch_name)
        
        if not branch_context:
            # Create new branch context
            context_data = {
                "description": f"Branch context for {branch_name}",
                "last_session_id": current_session["id"],
                "push_prepared": True
            }
        else:
            context_data = branch_context.get("context_data", {})
            context_data["last_session_id"] = current_session["id"]
            context_data["push_prepared"] = True
        
        # Include session summary
        session_summary = {
            "session_id": current_session["id"],
            "created_at": current_session["created_at"],
            "updated_at": current_session["updated_at"],
            "tool_usage_count": len(current_session.get("tool_usage", [])),
            "file_interactions_count": len(current_session.get("file_interactions", [])),
            "status": current_session.get("status", "active")
        }
        
        # Update active sessions list
        active_sessions = branch_context.get("active_sessions", []) if branch_context else []
        if current_session["id"] not in active_sessions:
            active_sessions.append(current_session["id"])
        
        # Update branch metadata
        branch_metadata = branch_context.get("branch_metadata", {}) if branch_context else {}
        branch_metadata.update({
            "last_push_prep": datetime.now().isoformat(),
            "push_count": branch_metadata.get("push_count", 0) + 1,
            "last_session_summary": session_summary
        })
        
        # Update branch context in database
        return memory_db.update_branch_context(
            branch_name=branch_name,
            base_branch=branch_context.get("base_branch") if branch_context else None,
            context_data=context_data,
            active_sessions=active_sessions,
            memory_snapshot=current_session.get("context_data", {}),
            branch_metadata=branch_metadata
        )
        
    except Exception as e:
        logging.error(f"Failed to update branch context for push: {e}")
        return False

def validate_memory_consistency(memory_db: MemoryDatabase, 
                               git_extractor: GitMetadataExtractor) -> dict:
    """Validate memory system consistency before push."""
    try:
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        # Check database consistency
        try:
            recent_sessions = memory_db.get_recent_sessions(5)
            recent_commits = memory_db.get_recent_commits(limit=5)
            validation_results["recent_sessions_count"] = len(recent_sessions)
            validation_results["recent_commits_count"] = len(recent_commits)
        except Exception as e:
            validation_results["issues"].append(f"Database query failed: {e}")
            validation_results["valid"] = False
        
        # Check git repository state
        try:
            current_branch = git_extractor.get_current_branch()
            latest_commit = git_extractor.get_latest_commit_sha()
            
            if not current_branch:
                validation_results["warnings"].append("Could not determine current branch")
            
            if not latest_commit:
                validation_results["warnings"].append("Could not get latest commit SHA")
            
            validation_results["git_branch"] = current_branch
            validation_results["git_latest_commit"] = latest_commit[:8] if latest_commit else None
            
        except Exception as e:
            validation_results["issues"].append(f"Git state check failed: {e}")
            validation_results["valid"] = False
        
        # Check for orphaned sessions
        try:
            orphaned_count = check_orphaned_sessions(memory_db)
            if orphaned_count > 10:
                validation_results["warnings"].append(f"Found {orphaned_count} orphaned sessions")
            validation_results["orphaned_sessions"] = orphaned_count
        except Exception as e:
            validation_results["warnings"].append(f"Orphaned session check failed: {e}")
        
        # Check storage space
        try:
            storage_path = memory_dir / "storage"
            if storage_path.exists():
                total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                validation_results["storage_size_mb"] = round(size_mb, 2)
                
                if size_mb > 1000:  # 1GB warning
                    validation_results["warnings"].append(f"Memory storage is large: {size_mb:.1f}MB")
        except Exception as e:
            validation_results["warnings"].append(f"Storage check failed: {e}")
        
        validation_results["validation_time"] = datetime.now().isoformat()
        
        return validation_results
        
    except Exception as e:
        logging.error(f"Failed to validate memory consistency: {e}")
        return {"valid": False, "error": str(e)}

def check_orphaned_sessions(memory_db: MemoryDatabase) -> int:
    """Check for orphaned sessions without proper endings."""
    try:
        # Get sessions that are still marked as active but are old
        cutoff_time = datetime.now().timestamp() - (24 * 3600)  # 24 hours ago
        
        recent_sessions = memory_db.get_recent_sessions(100)
        orphaned_count = 0
        
        for session in recent_sessions:
            try:
                created_at = datetime.fromisoformat(session["created_at"]).timestamp()
                if (created_at < cutoff_time and 
                    session.get("status", "active") == "active" and
                    not session.get("ended_at")):
                    orphaned_count += 1
            except:
                continue
        
        return orphaned_count
        
    except Exception as e:
        logging.error(f"Failed to check orphaned sessions: {e}")
        return 0

def create_sync_metadata(git_extractor: GitMetadataExtractor, 
                        current_session: dict, 
                        branch_name: str) -> dict:
    """Create synchronization metadata."""
    try:
        # Get machine identifier
        machine_id = get_machine_id()
        
        # Get git state
        current_branch = git_extractor.get_current_branch()
        latest_commit = git_extractor.get_latest_commit_sha()
        
        metadata = {
            "machine_id": machine_id,
            "sync_timestamp": datetime.now().isoformat(),
            "git_branch": current_branch,
            "target_branch": branch_name,
            "latest_commit": latest_commit,
            "session_id": current_session["id"] if current_session else None,
            "sync_type": "pre_push",
            "memory_system_version": "2.0"
        }
        
        # Add session context if available
        if current_session:
            metadata["session_context"] = {
                "created_at": current_session["created_at"],
                "updated_at": current_session["updated_at"],
                "tool_usage_count": len(current_session.get("tool_usage", [])),
                "file_interactions_count": len(current_session.get("file_interactions", [])),
                "checkpoints_count": len(current_session.get("checkpoints", []))
            }
        
        return metadata
        
    except Exception as e:
        logging.error(f"Failed to create sync metadata: {e}")
        return {"error": str(e)}

def store_sync_metadata(memory_db: MemoryDatabase, sync_metadata: dict) -> bool:
    """Store synchronization metadata in database."""
    try:
        if "error" in sync_metadata:
            return False
        
        # Store in git_sync_status table
        with memory_db._get_connection() as conn:
            conn.execute("""
                INSERT INTO git_sync_status 
                (machine_id, branch_name, last_sync_commit, sync_timestamp, 
                 sync_status, conflict_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                sync_metadata.get("machine_id"),
                sync_metadata.get("target_branch"),
                sync_metadata.get("latest_commit"),
                sync_metadata.get("sync_timestamp"),
                "completed",
                json.dumps(sync_metadata)
            ))
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to store sync metadata: {e}")
        return False

def get_machine_id() -> str:
    """Get a machine identifier."""
    try:
        import socket
        import hashlib
        
        # Use hostname and try to get a stable identifier
        hostname = socket.gethostname()
        
        # Try to get MAC address for more stability
        try:
            import uuid
            mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
            identifier = f"{hostname}_{mac}"
        except:
            identifier = hostname
        
        # Create a short hash
        return hashlib.md5(identifier.encode()).hexdigest()[:8]
        
    except Exception as e:
        logging.error(f"Failed to get machine ID: {e}")
        return "unknown"

def main():
    """Main hook execution."""
    try:
        if len(sys.argv) < 2:
            print("Usage: git-memory-sync.py <branch_name>", file=sys.stderr)
            sys.exit(0)
        
        branch_name = sys.argv[1]
        
        # Perform memory sync
        result = sync_memory_before_push(branch_name)
        
        if not result["success"]:
            print(f"Warning: Memory sync failed: {result.get('error', 'Unknown error')}", 
                  file=sys.stderr)
            # Don't block push on sync failure
        else:
            # Check for validation warnings
            validation = result.get("validation", {})
            warnings = validation.get("warnings", [])
            
            if warnings:
                print("Memory sync warnings:", file=sys.stderr)
                for warning in warnings:
                    print(f"  - {warning}", file=sys.stderr)
            
            if not validation.get("valid", True):
                print("Memory validation failed - push proceeding but memory may be inconsistent", 
                      file=sys.stderr)
        
        # Exit successfully (don't block push)
        sys.exit(0)
        
    except Exception as e:
        print(f"Memory sync hook error: {e}", file=sys.stderr)
        # Don't block push on hook failure
        sys.exit(0)

if __name__ == "__main__":
    main()