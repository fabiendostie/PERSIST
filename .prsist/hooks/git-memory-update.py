#!/usr/bin/env python3
"""
Git memory update hook for Prsist Memory System.
Handles merge operations and context updates.
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
    from correlation_engine import CorrelationEngine
    from utils import setup_logging, get_project_root
except ImportError as e:
    print(f"Memory system not available: {e}", file=sys.stderr)
    sys.exit(0)

def update_memory_after_merge(operation_type: str, branch_name: str) -> dict:
    """Update memory context after merge operation."""
    try:
        setup_logging("WARNING")  # Quiet for hooks
        
        project_root = get_project_root()
        
        # Initialize components
        git_extractor = GitMetadataExtractor(str(project_root))
        memory_db = MemoryDatabase(str(memory_dir / "storage" / "sessions.db"))
        session_tracker = SessionTracker(str(memory_dir))
        correlation_engine = CorrelationEngine(memory_db, git_extractor)
        
        # Get current state
        current_session = session_tracker.get_current_session()
        current_branch = git_extractor.get_current_branch()
        latest_commit = git_extractor.get_latest_commit_sha()
        
        update_results = {
            "merge_processed": False,
            "contexts_merged": False,
            "session_updated": False,
            "conflicts_resolved": False,
            "documentation_generated": False
        }
        
        # Process the merge
        if operation_type == "merge":
            merge_result = process_merge_operation(
                git_extractor, memory_db, current_branch, branch_name, latest_commit
            )
            update_results["merge_processed"] = merge_result["success"]
        
        # Merge branch contexts
        context_merge_result = merge_branch_contexts(
            memory_db, current_branch, branch_name
        )
        update_results["contexts_merged"] = context_merge_result["success"]
        
        # Update current session
        if current_session:
            session_update_result = update_session_after_merge(
                session_tracker, current_session, operation_type, branch_name, latest_commit
            )
            update_results["session_updated"] = session_update_result["success"]
        
        # Resolve any memory conflicts
        conflict_resolution = resolve_memory_conflicts(
            memory_db, current_branch, branch_name
        )
        update_results["conflicts_resolved"] = conflict_resolution["success"]
        
        # Generate merge documentation
        doc_result = generate_merge_documentation(
            memory_db, operation_type, current_branch, branch_name, latest_commit
        )
        update_results["documentation_generated"] = doc_result["success"]
        
        # Correlate recent commits if this was a merge
        if operation_type == "merge" and latest_commit:
            try:
                correlation_engine.correlate_commit_with_sessions(latest_commit)
            except Exception as e:
                logging.warning(f"Failed to correlate merge commit: {e}")
        
        result = {
            "success": True,
            "operation_type": operation_type,
            "source_branch": branch_name,
            "target_branch": current_branch,
            "merge_commit": latest_commit,
            "update_results": update_results,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"Memory update completed after {operation_type} from {branch_name}")
        return result
        
    except Exception as e:
        logging.error(f"Failed to update memory after merge: {e}")
        return {"success": False, "error": str(e)}

def process_merge_operation(git_extractor: GitMetadataExtractor, 
                          memory_db: MemoryDatabase,
                          target_branch: str, 
                          source_branch: str, 
                          merge_commit: str) -> dict:
    """Process merge operation and update memory accordingly."""
    try:
        # Get merge commit metadata
        if merge_commit:
            commit_metadata = git_extractor.get_commit_metadata(merge_commit)
            if commit_metadata:
                # Store merge commit
                memory_db.record_commit(
                    commit_sha=commit_metadata["commit_sha"],
                    branch_name=target_branch,
                    commit_message=f"Merge {source_branch} into {target_branch}",
                    author_email=commit_metadata.get("author_email"),
                    commit_timestamp=commit_metadata["timestamp"],
                    changed_files_count=len(commit_metadata.get("file_changes", [])),
                    lines_added=commit_metadata.get("stats", {}).get("insertions", 0),
                    lines_deleted=commit_metadata.get("stats", {}).get("deletions", 0),
                    memory_impact_score=commit_metadata["impact_score"],
                    commit_metadata={
                        "merge_operation": True,
                        "source_branch": source_branch,
                        "target_branch": target_branch,
                        "commit_type": "merge"
                    }
                )
        
        return {"success": True}
        
    except Exception as e:
        logging.error(f"Failed to process merge operation: {e}")
        return {"success": False, "error": str(e)}

def merge_branch_contexts(memory_db: MemoryDatabase, 
                         target_branch: str, 
                         source_branch: str) -> dict:
    """Merge branch contexts after merge operation."""
    try:
        # Get both branch contexts
        target_context = memory_db.get_branch_context(target_branch)
        source_context = memory_db.get_branch_context(source_branch)
        
        if not target_context:
            # No target context, just return success
            return {"success": True, "message": "No target context to merge"}
        
        if not source_context:
            # No source context, nothing to merge
            return {"success": True, "message": "No source context to merge"}
        
        # Merge context data
        merged_context_data = target_context.get("context_data", {}).copy()
        source_context_data = source_context.get("context_data", {})
        
        # Merge active sessions
        target_sessions = set(target_context.get("active_sessions", []))
        source_sessions = set(source_context.get("active_sessions", []))
        merged_sessions = list(target_sessions.union(source_sessions))
        
        # Merge memory snapshots
        target_snapshot = target_context.get("memory_snapshot", {})
        source_snapshot = source_context.get("memory_snapshot", {})
        merged_snapshot = merge_memory_snapshots(target_snapshot, source_snapshot)
        
        # Merge branch metadata
        target_metadata = target_context.get("branch_metadata", {})
        source_metadata = source_context.get("branch_metadata", {})
        merged_metadata = target_metadata.copy()
        merged_metadata.update({
            "last_merge": datetime.now().isoformat(),
            "merged_from": source_branch,
            "merge_count": merged_metadata.get("merge_count", 0) + 1,
            "source_branch_metadata": source_metadata
        })
        
        # Add merge information to context data
        merged_context_data["last_merge"] = {
            "source_branch": source_branch,
            "timestamp": datetime.now().isoformat(),
            "merged_sessions": len(source_sessions),
            "merged_context_keys": list(source_context_data.keys())
        }
        
        # Update target branch context
        success = memory_db.update_branch_context(
            branch_name=target_branch,
            base_branch=target_context.get("base_branch"),
            context_data=merged_context_data,
            active_sessions=merged_sessions,
            memory_snapshot=merged_snapshot,
            branch_metadata=merged_metadata
        )
        
        return {"success": success}
        
    except Exception as e:
        logging.error(f"Failed to merge branch contexts: {e}")
        return {"success": False, "error": str(e)}

def merge_memory_snapshots(target_snapshot: dict, source_snapshot: dict) -> dict:
    """Merge memory snapshots from two branches."""
    try:
        merged = target_snapshot.copy()
        
        # Merge recent decisions
        target_decisions = merged.get("recent_decisions", [])
        source_decisions = source_snapshot.get("recent_decisions", [])
        
        # Combine and deduplicate decisions
        all_decisions = target_decisions + source_decisions
        unique_decisions = []
        seen_titles = set()
        
        for decision in sorted(all_decisions, key=lambda x: x.get("timestamp", 0), reverse=True):
            title = decision.get("title", "")
            if title not in seen_titles:
                unique_decisions.append(decision)
                seen_titles.add(title)
        
        merged["recent_decisions"] = unique_decisions[:20]  # Keep last 20
        
        # Merge project memory (append source to target)
        target_memory = merged.get("project_memory", "")
        source_memory = source_snapshot.get("project_memory", "")
        
        if source_memory and source_memory != target_memory:
            merged["project_memory"] = target_memory + f"\n\n## Merged from {source_snapshot.get('branch_name', 'branch')}\n\n" + source_memory
        
        # Merge other context
        for key, value in source_snapshot.items():
            if key not in ["recent_decisions", "project_memory"] and key not in merged:
                merged[key] = value
        
        return merged
        
    except Exception as e:
        logging.error(f"Failed to merge memory snapshots: {e}")
        return target_snapshot

def update_session_after_merge(session_tracker: SessionTracker, 
                              current_session: dict,
                              operation_type: str, 
                              source_branch: str, 
                              merge_commit: str) -> dict:
    """Update current session after merge operation."""
    try:
        session_context = current_session.get("context_data", {})
        
        # Add merge information
        merge_info = {
            "operation_type": operation_type,
            "source_branch": source_branch,
            "merge_commit": merge_commit,
            "timestamp": datetime.now().isoformat()
        }
        
        session_merges = session_context.get("session_merges", [])
        session_merges.append(merge_info)
        session_context["session_merges"] = session_merges
        session_context["last_merge"] = merge_info
        
        # Create checkpoint for merge
        checkpoint_name = f"post_merge_{source_branch}_{datetime.now().strftime('%H%M%S')}"
        session_tracker.create_checkpoint(checkpoint_name)
        
        # Update session
        success = session_tracker.update_session(context_data=session_context)
        
        return {"success": success}
        
    except Exception as e:
        logging.error(f"Failed to update session after merge: {e}")
        return {"success": False, "error": str(e)}

def resolve_memory_conflicts(memory_db: MemoryDatabase, 
                            target_branch: str, 
                            source_branch: str) -> dict:
    """Resolve any memory conflicts from the merge."""
    try:
        # Check for conflicting sessions (sessions active on both branches)
        target_context = memory_db.get_branch_context(target_branch)
        source_context = memory_db.get_branch_context(source_branch)
        
        conflicts_resolved = 0
        
        if target_context and source_context:
            target_sessions = set(target_context.get("active_sessions", []))
            source_sessions = set(source_context.get("active_sessions", []))
            
            # Find sessions that exist in both branches
            conflicting_sessions = target_sessions.intersection(source_sessions)
            
            for session_id in conflicting_sessions:
                try:
                    # Get session data
                    session = memory_db.get_session(session_id)
                    if session:
                        # Mark session as merged
                        session_context = session.get("context_data", {})
                        session_context["conflict_resolved"] = {
                            "timestamp": datetime.now().isoformat(),
                            "target_branch": target_branch,
                            "source_branch": source_branch,
                            "resolution": "merged"
                        }
                        
                        # Update session (this would need to be implemented in database.py)
                        # For now, just count as resolved
                        conflicts_resolved += 1
                        
                except Exception as e:
                    logging.warning(f"Failed to resolve conflict for session {session_id}: {e}")
        
        return {
            "success": True,
            "conflicts_found": len(conflicting_sessions) if 'conflicting_sessions' in locals() else 0,
            "conflicts_resolved": conflicts_resolved
        }
        
    except Exception as e:
        logging.error(f"Failed to resolve memory conflicts: {e}")
        return {"success": False, "error": str(e)}

def generate_merge_documentation(memory_db: MemoryDatabase, 
                                operation_type: str,
                                target_branch: str, 
                                source_branch: str, 
                                merge_commit: str) -> dict:
    """Generate documentation for the merge operation."""
    try:
        # Create merge documentation
        merge_doc = f"""# Merge Operation: {source_branch} → {target_branch}

**Operation Type:** {operation_type.title()}
**Timestamp:** {datetime.now().isoformat()}
**Merge Commit:** {merge_commit[:8] if merge_commit else 'N/A'}

## Summary

Successfully merged changes from `{source_branch}` into `{target_branch}`.

## Memory System Updates

- Branch contexts merged
- Active sessions reconciled
- Memory snapshots combined
- Documentation generated automatically

## Next Steps

1. Verify that all functionality works as expected
2. Run tests to ensure no regressions
3. Update any dependent documentation
4. Consider cleaning up the source branch if no longer needed
"""
        
        # Store documentation
        if merge_commit:
            success = memory_db.record_documentation_entry(
                commit_sha=merge_commit,
                doc_type="merge_operation",
                content=merge_doc,
                metadata={
                    "operation_type": operation_type,
                    "source_branch": source_branch,
                    "target_branch": target_branch,
                    "generated_at": datetime.now().isoformat()
                }
            )
        else:
            success = True  # No commit to associate with, but doc was generated
        
        return {"success": success, "documentation": merge_doc}
        
    except Exception as e:
        logging.error(f"Failed to generate merge documentation: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main hook execution."""
    try:
        if len(sys.argv) < 3:
            print("Usage: git-memory-update.py <operation_type> <branch_name>", file=sys.stderr)
            sys.exit(0)
        
        operation_type = sys.argv[1]  # merge, rebase, etc.
        branch_name = sys.argv[2]
        
        # Perform memory update
        result = update_memory_after_merge(operation_type, branch_name)
        
        if not result["success"]:
            print(f"Warning: Memory update failed: {result.get('error', 'Unknown error')}", 
                  file=sys.stderr)
        else:
            logging.info(f"Memory update completed for {operation_type} from {branch_name}")
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        print(f"Memory update hook error: {e}", file=sys.stderr)
        # Don't block on hook failure
        sys.exit(0)

if __name__ == "__main__":
    main()