#!/usr/bin/env python3
"""
Git context switch hook for Prsist Memory System.
Switches memory context when changing branches.
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
    from context_builder import ContextBuilder
    from utils import setup_logging, get_project_root
except ImportError as e:
    print(f"Memory system not available: {e}", file=sys.stderr)
    sys.exit(0)

def switch_branch_context(branch_name: str) -> dict:
    """Switch memory context for branch change."""
    try:
        setup_logging("WARNING")  # Quiet for hooks
        
        project_root = get_project_root()
        
        # Initialize components
        git_extractor = GitMetadataExtractor(str(project_root))
        memory_db = MemoryDatabase(str(memory_dir / "storage" / "sessions.db"))
        session_tracker = SessionTracker(str(memory_dir))
        context_builder = ContextBuilder(str(memory_dir))
        
        # Get current branch info
        current_branch = git_extractor.get_current_branch()
        
        # Save current branch context if we have an active session
        current_session = session_tracker.get_current_session()
        if current_session:
            save_result = save_current_branch_context(
                memory_db, current_session, current_branch
            )
            if not save_result["success"]:
                logging.warning(f"Failed to save current branch context: {save_result.get('error')}")
        
        # Load context for new branch
        branch_context = load_branch_context(memory_db, branch_name)
        
        # Update current session with branch context
        if current_session:
            update_result = update_session_for_branch(
                session_tracker, current_session, branch_name, branch_context
            )
            if not update_result["success"]:
                logging.warning(f"Failed to update session for branch: {update_result.get('error')}")
        
        # Create checkpoint for branch switch
        if current_session:
            checkpoint_name = f"branch_switch_{branch_name}_{datetime.now().strftime('%H%M%S')}"
            session_tracker.create_checkpoint(checkpoint_name)
        
        result = {
            "success": True,
            "previous_branch": current_branch,
            "new_branch": branch_name,
            "branch_context": branch_context,
            "session_updated": current_session is not None
        }
        
        logging.info(f"Switched context from {current_branch} to {branch_name}")
        return result
        
    except Exception as e:
        logging.error(f"Failed to switch branch context: {e}")
        return {"success": False, "error": str(e)}

def save_current_branch_context(memory_db: MemoryDatabase, 
                               current_session: dict, 
                               branch_name: str) -> dict:
    """Save current branch context."""
    try:
        # Prepare branch context data
        context_data = {
            "last_session_id": current_session["id"],
            "last_updated": datetime.now().isoformat(),
            "session_summary": {
                "created_at": current_session["created_at"],
                "updated_at": current_session["updated_at"],
                "tool_usage_count": len(current_session.get("tool_usage", [])),
                "file_interactions_count": len(current_session.get("file_interactions", [])),
                "checkpoints_count": len(current_session.get("checkpoints", []))
            }
        }
        
        # Include recent commits if any
        session_context = current_session.get("context_data", {})
        session_commits = session_context.get("session_commits", [])
        if session_commits:
            context_data["recent_commits"] = session_commits[-5:]  # Last 5 commits
        
        # Get base branch information
        base_branch = determine_base_branch(branch_name)
        
        # Update branch context in database
        success = memory_db.update_branch_context(
            branch_name=branch_name,
            base_branch=base_branch,
            context_data=context_data,
            active_sessions=[current_session["id"]],
            memory_snapshot=session_context,
            branch_metadata={
                "last_switch_time": datetime.now().isoformat(),
                "switch_count": get_branch_switch_count(memory_db, branch_name) + 1
            }
        )
        
        if success:
            return {"success": True, "context_saved": True}
        else:
            return {"success": False, "error": "Failed to update database"}
        
    except Exception as e:
        logging.error(f"Failed to save branch context: {e}")
        return {"success": False, "error": str(e)}

def load_branch_context(memory_db: MemoryDatabase, branch_name: str) -> dict:
    """Load context for the target branch."""
    try:
        # Get existing branch context
        branch_context = memory_db.get_branch_context(branch_name)
        
        if not branch_context:
            # No existing context, create default
            return create_default_branch_context(memory_db, branch_name)
        
        # Enhance context with recent information
        enhanced_context = enhance_branch_context(memory_db, branch_context)
        
        return enhanced_context
        
    except Exception as e:
        logging.error(f"Failed to load branch context: {e}")
        return {"error": str(e)}

def create_default_branch_context(memory_db: MemoryDatabase, branch_name: str) -> dict:
    """Create default context for a new branch."""
    try:
        base_branch = determine_base_branch(branch_name)
        
        default_context = {
            "branch_name": branch_name,
            "base_branch": base_branch,
            "created_at": datetime.now().isoformat(),
            "context_data": {
                "description": f"New branch context for {branch_name}",
                "goals": [],
                "related_issues": [],
                "development_notes": []
            },
            "active_sessions": [],
            "memory_snapshot": {},
            "branch_metadata": {
                "created_time": datetime.now().isoformat(),
                "switch_count": 1,
                "branch_type": classify_branch_type(branch_name)
            },
            "is_new": True
        }
        
        # Try to inherit context from base branch
        if base_branch and base_branch != branch_name:
            base_context = memory_db.get_branch_context(base_branch)
            if base_context:
                # Inherit relevant information
                default_context["context_data"]["inherited_from"] = base_branch
                base_memory = base_context.get("memory_snapshot", {})
                if base_memory:
                    default_context["context_data"]["base_context"] = {
                        "project_memory": base_memory.get("project_memory", ""),
                        "recent_decisions": base_memory.get("recent_decisions", [])
                    }
        
        return default_context
        
    except Exception as e:
        logging.error(f"Failed to create default branch context: {e}")
        return {"error": str(e)}

def enhance_branch_context(memory_db: MemoryDatabase, branch_context: dict) -> dict:
    """Enhance existing branch context with recent information."""
    try:
        branch_name = branch_context["branch_name"]
        
        # Get recent commits for this branch
        recent_commits = memory_db.get_recent_commits(branch_name=branch_name, limit=10)
        
        # Get recent sessions associated with this branch
        active_sessions = branch_context.get("active_sessions", [])
        session_summaries = []
        
        for session_id in active_sessions[-5:]:  # Last 5 sessions
            try:
                session = memory_db.get_session(session_id)
                if session:
                    session_summaries.append({
                        "id": session_id,
                        "created_at": session["created_at"],
                        "status": session.get("status", "unknown")
                    })
            except:
                continue  # Skip invalid sessions
        
        # Enhance context
        enhanced = branch_context.copy()
        enhanced["recent_commits"] = recent_commits
        enhanced["recent_sessions"] = session_summaries
        enhanced["last_loaded"] = datetime.now().isoformat()
        
        # Update activity metrics
        if "branch_metadata" not in enhanced:
            enhanced["branch_metadata"] = {}
        
        enhanced["branch_metadata"]["last_access"] = datetime.now().isoformat()
        enhanced["branch_metadata"]["commit_count"] = len(recent_commits)
        enhanced["branch_metadata"]["session_count"] = len(session_summaries)
        
        return enhanced
        
    except Exception as e:
        logging.error(f"Failed to enhance branch context: {e}")
        return branch_context

def update_session_for_branch(session_tracker: SessionTracker, 
                             current_session: dict, 
                             branch_name: str, 
                             branch_context: dict) -> dict:
    """Update current session with branch context."""
    try:
        # Prepare context updates
        session_context = current_session.get("context_data", {})
        
        # Update git information
        session_context["current_branch"] = branch_name
        session_context["branch_switch_time"] = datetime.now().isoformat()
        
        # Add branch context information
        if not branch_context.get("error"):
            session_context["branch_context"] = {
                "branch_name": branch_name,
                "base_branch": branch_context.get("base_branch"),
                "branch_type": branch_context.get("branch_metadata", {}).get("branch_type"),
                "is_new_branch": branch_context.get("is_new", False)
            }
            
            # Include recent commits context
            recent_commits = branch_context.get("recent_commits", [])
            if recent_commits:
                session_context["branch_context"]["recent_commits"] = [
                    {
                        "sha": commit["commit_sha"][:8],
                        "message": commit.get("commit_message", ""),
                        "timestamp": commit.get("commit_timestamp")
                    }
                    for commit in recent_commits[:3]  # Last 3 commits
                ]
            
            # Include branch goals and notes
            branch_data = branch_context.get("context_data", {})
            if branch_data.get("goals"):
                session_context["branch_context"]["goals"] = branch_data["goals"]
            if branch_data.get("development_notes"):
                session_context["branch_context"]["notes"] = branch_data["development_notes"]
        
        # Update session
        success = session_tracker.update_session(context_data=session_context)
        
        return {"success": success}
        
    except Exception as e:
        logging.error(f"Failed to update session for branch: {e}")
        return {"success": False, "error": str(e)}

def determine_base_branch(branch_name: str) -> str:
    """Determine the base branch for a given branch."""
    # Common patterns
    if branch_name in ["main", "master", "develop", "dev"]:
        return None  # These are base branches
    
    # Feature branches
    if branch_name.startswith(("feature/", "feat/")):
        return "develop" if "develop" in branch_name else "main"
    
    # Bugfix branches
    if branch_name.startswith(("bugfix/", "fix/", "hotfix/")):
        return "main"
    
    # Release branches
    if branch_name.startswith("release/"):
        return "develop"
    
    # Default to main
    return "main"

def classify_branch_type(branch_name: str) -> str:
    """Classify the type of branch."""
    if branch_name in ["main", "master"]:
        return "main"
    elif branch_name in ["develop", "dev"]:
        return "develop"
    elif branch_name.startswith(("feature/", "feat/")):
        return "feature"
    elif branch_name.startswith(("bugfix/", "fix/")):
        return "bugfix"
    elif branch_name.startswith("hotfix/"):
        return "hotfix"
    elif branch_name.startswith("release/"):
        return "release"
    elif branch_name.startswith(("chore/", "docs/", "test/")):
        return "maintenance"
    else:
        return "other"

def get_branch_switch_count(memory_db: MemoryDatabase, branch_name: str) -> int:
    """Get the number of times this branch has been switched to."""
    try:
        branch_context = memory_db.get_branch_context(branch_name)
        if branch_context:
            return branch_context.get("branch_metadata", {}).get("switch_count", 0)
        return 0
    except:
        return 0

def main():
    """Main hook execution."""
    try:
        if len(sys.argv) < 2:
            print("Usage: git-context-switch.py <branch_name>", file=sys.stderr)
            sys.exit(0)
        
        branch_name = sys.argv[1]
        
        # Skip for HEAD and other special refs
        if branch_name in ["HEAD", ""] or branch_name.startswith("refs/"):
            sys.exit(0)
        
        # Perform context switch
        result = switch_branch_context(branch_name)
        
        if not result["success"]:
            print(f"Warning: Failed to switch branch context: {result.get('error', 'Unknown error')}", 
                  file=sys.stderr)
        else:
            logging.info(f"Successfully switched to branch {branch_name}")
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        print(f"Context switch hook error: {e}", file=sys.stderr)
        # Don't block on hook failure
        sys.exit(0)

if __name__ == "__main__":
    main()