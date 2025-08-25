#!/usr/bin/env python3
"""
Enhanced Git Integration for Prsist Memory System - Phase 2.
Provides automatic commit correlation, branch context management, and session tracking.
"""

import os
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from git_integration import GitMetadataExtractor, ChangeImpactAnalyzer
from database import MemoryDatabase


class EnhancedGitIntegrator:
    """Enhanced git integration with automatic commit correlation and branch management."""
    
    def __init__(self, memory_dir: str, repo_path: str = "."):
        """Initialize enhanced git integrator."""
        self.memory_dir = Path(memory_dir)
        self.repo_path = Path(repo_path).resolve()
        
        # Initialize components
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        self.git_extractor = GitMetadataExtractor(str(self.repo_path))
        self.change_analyzer = ChangeImpactAnalyzer(self.git_extractor)
        
        # Track current state
        self.current_branch = None
        self.current_commit = None
        self.last_correlation_check = None
        
        self._update_current_state()
        
        logging.info(f"Enhanced Git Integrator initialized for {self.repo_path}")
    
    def _update_current_state(self) -> None:
        """Update current git state tracking."""
        try:
            self.current_branch = self.git_extractor.get_current_branch()
            self.current_commit = self.git_extractor.get_latest_commit_sha()
            self.last_correlation_check = datetime.now()
        except Exception as e:
            logging.error(f"Failed to update git state: {e}")
    
    def auto_correlate_session(self, session_id: str) -> Dict[str, Any]:
        """Automatically correlate session with current git state."""
        try:
            self._update_current_state()
            
            if not self.current_commit:
                return {"correlated": False, "reason": "No git commits found"}
            
            # Get commit metadata
            commit_metadata = self.git_extractor.get_commit_metadata(self.current_commit)
            if not commit_metadata:
                return {"correlated": False, "reason": "Failed to extract commit metadata"}
            
            # Analyze commit impact
            impact_analysis = self.change_analyzer.analyze_commit_impact(commit_metadata)
            
            # Store git commit information
            success = self.db.record_commit(
                commit_sha=self.current_commit,
                session_id=session_id,
                branch_name=self.current_branch,
                commit_message=commit_metadata.get("subject"),
                author_email=commit_metadata.get("author_email"),
                commit_timestamp=commit_metadata.get("timestamp"),
                changed_files_count=commit_metadata.get("stats", {}).get("files_changed", 0),
                lines_added=commit_metadata.get("stats", {}).get("insertions", 0),
                lines_deleted=commit_metadata.get("stats", {}).get("deletions", 0),
                memory_impact_score=commit_metadata.get("impact_score", 0.0),
                commit_metadata={
                    "commit_type": commit_metadata.get("commit_type"),
                    "impact_analysis": impact_analysis,
                    "branches": commit_metadata.get("branches", [])
                }
            )
            
            if success:
                # Store file changes
                for file_change in commit_metadata.get("file_changes", []):
                    self.db.record_file_change(
                        commit_sha=self.current_commit,
                        session_id=session_id,
                        file_path=file_change.get("file_path"),
                        change_type=file_change.get("change_type"),
                        lines_added=file_change.get("lines_added", 0),
                        lines_deleted=file_change.get("lines_deleted", 0),
                        significance_score=file_change.get("significance_score", 0.0),
                        context_summary=f"{file_change.get('file_type')} file - {file_change.get('change_type')}"
                    )
                
                # Update branch context
                self.update_branch_context(self.current_branch, session_id)
                
                correlation_data = {
                    "correlated": True,
                    "commit_sha": self.current_commit,
                    "branch_name": self.current_branch,
                    "commit_metadata": commit_metadata,
                    "impact_analysis": impact_analysis,
                    "files_changed": len(commit_metadata.get("file_changes", [])),
                    "correlation_timestamp": datetime.now().isoformat()
                }
                
                logging.info(f"Auto-correlated session {session_id} with commit {self.current_commit[:8]}")
                return correlation_data
            else:
                return {"correlated": False, "reason": "Failed to store commit data"}
                
        except Exception as e:
            logging.error(f"Failed to auto-correlate session: {e}")
            return {"correlated": False, "reason": str(e)}
    
    def update_branch_context(self, branch_name: str, session_id: str) -> bool:
        """Update branch context with current session."""
        try:
            # Get recent commits on this branch
            recent_commits = self.get_branch_commits(branch_name, limit=5)
            
            # Build context summary
            context_summary = {
                "active_session": session_id,
                "recent_commits": len(recent_commits),
                "last_update": datetime.now().isoformat(),
                "commit_types": [c.get("commit_type", "unknown") for c in recent_commits]
            }
            
            # Store in database
            success = self.db.record_branch_context(
                branch_name=branch_name,
                commit_sha=self.current_commit,
                context_data=context_summary,
                session_id=session_id
            )
            
            if success:
                logging.info(f"Updated branch context for {branch_name}")
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to update branch context: {e}")
            return False
    
    def get_branch_commits(self, branch_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commits for a branch with metadata."""
        try:
            # Get commit SHAs for the branch
            cmd = ["git", "log", f"{branch_name}", "--format=%H", f"-{limit}"]
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            commit_shas = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            # Get metadata for each commit
            commits = []
            for sha in commit_shas:
                metadata = self.git_extractor.get_commit_metadata(sha)
                if metadata:
                    commits.append(metadata)
            
            return commits
            
        except Exception as e:
            logging.error(f"Failed to get branch commits: {e}")
            return []
    
    def switch_branch_context(self, from_branch: str, to_branch: str, session_id: str) -> Dict[str, Any]:
        """Handle branch context switching."""
        try:
            # Get context from target branch
            branch_context = self.db.get_branch_context(to_branch)
            previous_sessions = self.get_branch_sessions(to_branch, limit=5)
            
            # Analyze context differences
            context_diff = self.analyze_branch_context_diff(from_branch, to_branch)
            
            # Update current branch tracking
            self.current_branch = to_branch
            self.update_branch_context(to_branch, session_id)
            
            switch_data = {
                "from_branch": from_branch,
                "to_branch": to_branch,
                "branch_context": branch_context,
                "previous_sessions": len(previous_sessions),
                "context_differences": context_diff,
                "switch_timestamp": datetime.now().isoformat(),
                "recommendations": self.generate_branch_switch_recommendations(to_branch, previous_sessions)
            }
            
            logging.info(f"Switched branch context from {from_branch} to {to_branch}")
            return switch_data
            
        except Exception as e:
            logging.error(f"Failed to switch branch context: {e}")
            return {"error": str(e)}
    
    def analyze_branch_context_diff(self, branch1: str, branch2: str) -> Dict[str, Any]:
        """Analyze differences between branch contexts."""
        try:
            # Get commits unique to each branch
            cmd1 = ["git", "log", f"{branch1}..{branch2}", "--format=%H", "--max-count=20"]
            cmd2 = ["git", "log", f"{branch2}..{branch1}", "--format=%H", "--max-count=20"]
            
            result1 = subprocess.run(cmd1, cwd=self.repo_path, capture_output=True, text=True)
            result2 = subprocess.run(cmd2, cwd=self.repo_path, capture_output=True, text=True)
            
            commits_in_branch2 = result1.stdout.strip().split('\n') if result1.returncode == 0 else []
            commits_in_branch1 = result2.stdout.strip().split('\n') if result2.returncode == 0 else []
            
            # Clean empty strings
            commits_in_branch2 = [c for c in commits_in_branch2 if c.strip()]
            commits_in_branch1 = [c for c in commits_in_branch1 if c.strip()]
            
            return {
                "commits_ahead": len(commits_in_branch2),
                "commits_behind": len(commits_in_branch1),
                "diverged": len(commits_in_branch1) > 0 and len(commits_in_branch2) > 0,
                "up_to_date": len(commits_in_branch1) == 0 and len(commits_in_branch2) == 0
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze branch diff: {e}")
            return {"error": str(e)}
    
    def get_branch_sessions(self, branch_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sessions associated with a branch."""
        try:
            return self.db.get_branch_sessions(branch_name, limit)
        except Exception as e:
            logging.error(f"Failed to get branch sessions: {e}")
            return []
    
    def generate_branch_switch_recommendations(self, branch_name: str, previous_sessions: List[Dict]) -> List[str]:
        """Generate recommendations when switching branches."""
        recommendations = []
        
        if not previous_sessions:
            recommendations.append("This is a new branch or no previous memory sessions found")
            recommendations.append("Consider creating a checkpoint before making changes")
        else:
            recent_session = previous_sessions[0] if previous_sessions else None
            if recent_session:
                recommendations.append(f"Last memory session was {recent_session.get('days_ago', 'unknown')} days ago")
                recommendations.append("Review previous session context for continuity")
        
        # Check for uncommitted changes
        staged_files = self.git_extractor.get_staged_files()
        if staged_files:
            recommendations.append(f"You have {len(staged_files)} staged files - consider committing before switching context")
        
        return recommendations
    
    def get_commit_session_relationship(self, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get the relationship between a commit and memory sessions."""
        try:
            return self.db.get_commit_session_data(commit_sha)
        except Exception as e:
            logging.error(f"Failed to get commit session relationship: {e}")
            return None
    
    def track_merge_operation(self, merge_commit_sha: str, session_id: str) -> Dict[str, Any]:
        """Track a merge operation and its impact on memory."""
        try:
            merge_metadata = self.git_extractor.get_commit_metadata(merge_commit_sha)
            if not merge_metadata:
                return {"tracked": False, "reason": "Failed to get merge metadata"}
            
            # Analyze merge impact
            parent_commits = merge_metadata.get("parent_commits", [])
            if len(parent_commits) < 2:
                return {"tracked": False, "reason": "Not a merge commit"}
            
            # Get branches involved in merge
            merge_branches = []
            for parent in parent_commits:
                parent_branches = self.git_extractor.get_commit_branches(parent)
                merge_branches.extend(parent_branches)
            
            # Track the merge
            merge_data = {
                "merge_commit": merge_commit_sha,
                "parent_commits": parent_commits,
                "branches_merged": list(set(merge_branches)),
                "session_id": session_id,
                "merge_timestamp": datetime.now().isoformat(),
                "files_affected": len(merge_metadata.get("file_changes", [])),
                "impact_score": merge_metadata.get("impact_score", 0.0)
            }
            
            # Store merge information
            success = self.db.record_commit(
                commit_sha=merge_commit_sha,
                session_id=session_id,
                branch_name=self.current_branch,
                commit_message=merge_metadata.get("subject"),
                commit_metadata={"merge_operation": merge_data}
            )
            
            if success:
                logging.info(f"Tracked merge operation {merge_commit_sha[:8]}")
                return {"tracked": True, "merge_data": merge_data}
            else:
                return {"tracked": False, "reason": "Failed to store merge data"}
                
        except Exception as e:
            logging.error(f"Failed to track merge operation: {e}")
            return {"tracked": False, "reason": str(e)}
    
    def generate_git_memory_report(self, session_id: str = None, branch_name: str = None) -> Dict[str, Any]:
        """Generate comprehensive git-memory correlation report."""
        try:
            report = {
                "generation_time": datetime.now().isoformat(),
                "repository_path": str(self.repo_path),
                "current_state": {
                    "branch": self.current_branch,
                    "commit": self.current_commit,
                    "staged_files": len(self.git_extractor.get_staged_files())
                }
            }
            
            # Session-specific report
            if session_id:
                session_commits = self.db.get_session_git_commits(session_id)
                report["session_analysis"] = {
                    "session_id": session_id,
                    "commits_linked": len(session_commits),
                    "branches_touched": list(set(c.get("branch_name", "") for c in session_commits if c.get("branch_name"))),
                    "total_files_changed": sum(c.get("changed_files_count", 0) for c in session_commits),
                    "total_lines_added": sum(c.get("lines_added", 0) for c in session_commits),
                    "total_lines_deleted": sum(c.get("lines_deleted", 0) for c in session_commits)
                }
            
            # Branch-specific report
            if branch_name:
                branch_context = self.db.get_branch_context(branch_name)
                branch_sessions = self.get_branch_sessions(branch_name)
                report["branch_analysis"] = {
                    "branch_name": branch_name,
                    "context_available": branch_context is not None,
                    "associated_sessions": len(branch_sessions),
                    "recent_activity": branch_sessions[0] if branch_sessions else None
                }
            
            return report
            
        except Exception as e:
            logging.error(f"Failed to generate git memory report: {e}")
            return {"error": str(e)}
    
    def check_for_correlation_updates(self) -> Dict[str, Any]:
        """Check if git state has changed and needs correlation updates."""
        try:
            old_commit = self.current_commit
            old_branch = self.current_branch
            
            self._update_current_state()
            
            changes = {
                "commit_changed": old_commit != self.current_commit,
                "branch_changed": old_branch != self.current_branch,
                "needs_correlation": False,
                "changes_detected": []
            }
            
            if changes["commit_changed"]:
                changes["changes_detected"].append(f"New commit: {old_commit[:8] if old_commit else 'none'} -> {self.current_commit[:8] if self.current_commit else 'none'}")
                changes["needs_correlation"] = True
            
            if changes["branch_changed"]:
                changes["changes_detected"].append(f"Branch switch: {old_branch} -> {self.current_branch}")
                changes["needs_correlation"] = True
            
            return changes
            
        except Exception as e:
            logging.error(f"Failed to check for correlation updates: {e}")
            return {"error": str(e)}


class GitHookManager:
    """Manager for git hooks integration with memory system."""
    
    def __init__(self, repo_path: str, memory_dir: str):
        """Initialize git hook manager."""
        self.repo_path = Path(repo_path)
        self.memory_dir = Path(memory_dir)
        self.git_hooks_dir = self.repo_path / ".git" / "hooks"
        
    def install_memory_hooks(self) -> Dict[str, bool]:
        """Install git hooks for automatic memory correlation."""
        results = {}
        
        hooks = {
            "post-commit": self._generate_post_commit_hook(),
            "post-checkout": self._generate_post_checkout_hook(),
            "post-merge": self._generate_post_merge_hook()
        }
        
        for hook_name, hook_content in hooks.items():
            try:
                hook_path = self.git_hooks_dir / hook_name
                
                # Make backup if hook exists
                if hook_path.exists():
                    backup_path = hook_path.with_suffix(f"{hook_path.suffix}.backup")
                    hook_path.rename(backup_path)
                    logging.info(f"Backed up existing {hook_name} hook")
                
                # Write new hook
                hook_path.write_text(hook_content)
                hook_path.chmod(0o755)  # Make executable
                
                results[hook_name] = True
                logging.info(f"Installed {hook_name} hook")
                
            except Exception as e:
                logging.error(f"Failed to install {hook_name} hook: {e}")
                results[hook_name] = False
        
        return results
    
    def _generate_post_commit_hook(self) -> str:
        """Generate post-commit hook script."""
        return f"""#!/bin/sh
# Prsist Memory System - Post-Commit Hook
# Auto-correlate commits with active memory sessions

python3 "{self.memory_dir}/hooks/git-memory-correlate.py" post-commit "$@"
"""
    
    def _generate_post_checkout_hook(self) -> str:
        """Generate post-checkout hook script."""
        return f"""#!/bin/sh
# Prsist Memory System - Post-Checkout Hook
# Handle branch context switching

python3 "{self.memory_dir}/hooks/git-context-switch.py" post-checkout "$@"
"""
    
    def _generate_post_merge_hook(self) -> str:
        """Generate post-merge hook script."""
        return f"""#!/bin/sh
# Prsist Memory System - Post-Merge Hook
# Track merge operations and memory impact

python3 "{self.memory_dir}/hooks/git-memory-correlate.py" post-merge "$@"
"""
    
    def uninstall_memory_hooks(self) -> Dict[str, bool]:
        """Uninstall memory system git hooks."""
        results = {}
        hooks = ["post-commit", "post-checkout", "post-merge"]
        
        for hook_name in hooks:
            try:
                hook_path = self.git_hooks_dir / hook_name
                backup_path = hook_path.with_suffix(f"{hook_path.suffix}.backup")
                
                if hook_path.exists():
                    hook_path.unlink()
                    results[hook_name] = True
                    logging.info(f"Removed {hook_name} hook")
                    
                    # Restore backup if exists
                    if backup_path.exists():
                        backup_path.rename(hook_path)
                        logging.info(f"Restored backup for {hook_name} hook")
                else:
                    results[hook_name] = True  # Already uninstalled
                    
            except Exception as e:
                logging.error(f"Failed to uninstall {hook_name} hook: {e}")
                results[hook_name] = False
        
        return results


# CLI interface for enhanced git integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: enhanced_git_integration.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    memory_dir = os.environ.get("PRSIST_MEMORY_DIR", os.path.dirname(__file__))
    
    integrator = EnhancedGitIntegrator(memory_dir)
    
    if command == "auto-correlate":
        session_id = sys.argv[2] if len(sys.argv) > 2 else None
        if session_id:
            result = integrator.auto_correlate_session(session_id)
            print(json.dumps(result, indent=2))
        else:
            print("Error: Session ID required")
            sys.exit(1)
    
    elif command == "switch-context":
        if len(sys.argv) >= 4:
            from_branch, to_branch, session_id = sys.argv[2], sys.argv[3], sys.argv[4]
            result = integrator.switch_branch_context(from_branch, to_branch, session_id)
            print(json.dumps(result, indent=2))
        else:
            print("Error: from_branch, to_branch, and session_id required")
            sys.exit(1)
    
    elif command == "check-updates":
        result = integrator.check_for_correlation_updates()
        print(json.dumps(result, indent=2))
    
    elif command == "report":
        session_id = sys.argv[2] if len(sys.argv) > 2 else None
        branch_name = sys.argv[3] if len(sys.argv) > 3 else None
        result = integrator.generate_git_memory_report(session_id, branch_name)
        print(json.dumps(result, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)