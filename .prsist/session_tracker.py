#!/usr/bin/env python3
"""
Session tracking module for Prsist Memory System.
Manages session lifecycle and data collection.
"""

import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from database import MemoryDatabase
from utils import (
    get_git_info, 
    get_project_root, 
    save_json_file, 
    load_json_file,
    calculate_file_hash,
    sanitize_input
)

class SessionTracker:
    """Manages session tracking and lifecycle."""
    
    def __init__(self, memory_dir: str = None):
        """Initialize session tracker."""
        if memory_dir is None:
            memory_dir = Path(__file__).parent
        
        self.memory_dir = Path(memory_dir)
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        self.current_session_file = self.memory_dir / "sessions" / "active" / "current-session.json"
        self.project_root = get_project_root()
        
        # Ensure directories exist
        self.current_session_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start new session and return session context."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Get git information
        git_info = get_git_info(str(self.project_root))
        
        # Prepare session data
        session_data = {
            "id": session_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "project_path": str(self.project_root),
            "git_info": git_info,
            "context_data": sanitize_input(context_data) if context_data else {},
            "status": "active",
            "tool_usage": [],
            "file_interactions": [],
            "checkpoints": []
        }
        
        try:
            # Save to database
            self.db.create_session(
                session_id=session_id,
                project_path=str(self.project_root),
                context_data=session_data["context_data"],
                git_info=git_info
            )
            
            # Save current session file
            save_json_file(session_data, self.current_session_file)
            
            logging.info(f"Started new session: {session_id}")
            
            # Return context for Claude
            return {
                "session_id": session_id,
                "project_path": str(self.project_root),
                "git_info": git_info,
                "context_data": session_data["context_data"],
                "memory_system_active": True
            }
            
        except Exception as e:
            logging.error(f"Failed to start session: {e}")
            return {
                "session_id": None,
                "memory_system_active": False,
                "error": str(e)
            }
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current active session."""
        try:
            # First try the JSON file (for compatibility)
            if self.current_session_file.exists():
                return load_json_file(str(self.current_session_file))
            
            # Fall back to checking database for active sessions
            recent_sessions = self.db.get_recent_sessions(limit=10)
            
            for session_data in recent_sessions:
                if session_data.get("status") == "active":
                    # Convert to expected format
                    return {
                        "id": session_data["id"],
                        "created_at": session_data["created_at"],
                        "updated_at": session_data.get("updated_at"),
                        "project_path": session_data["project_path"],
                        "git_info": session_data.get("git_info", {}),
                        "context_data": session_data.get("context_data", {}),
                        "status": session_data["status"],
                        "tool_usage": [],
                        "file_interactions": [],
                        "checkpoints": []
                    }
            
            return None
        except Exception as e:
            logging.error(f"Failed to get current session: {e}")
            return None
    
    def update_session(self, **kwargs) -> bool:
        """Update current session with new data."""
        try:
            session_data = self.get_current_session()
            if not session_data:
                logging.warning("No active session to update")
                return False
            
            # Update session data
            session_data["updated_at"] = datetime.now().isoformat()
            
            # Handle specific updates
            for key, value in kwargs.items():
                if key in ["context_data", "status"]:
                    session_data[key] = sanitize_input(value)
                elif key == "git_info":
                    session_data["git_info"] = value
            
            # Save updates
            save_json_file(session_data, self.current_session_file)
            
            # Update database
            self.db.update_session(session_data["id"], **kwargs)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to update session: {e}")
            return False
    
    def log_tool_usage(self, tool_name: str, input_data: Any = None, 
                      output_data: Any = None, execution_time_ms: int = None,
                      success: bool = True) -> bool:
        """Log tool usage for current session."""
        try:
            session_data = self.get_current_session()
            if not session_data:
                logging.warning("No active session for tool usage logging")
                return False
            
            tool_entry = {
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "input_data": sanitize_input(input_data),
                "output_data": sanitize_input(output_data),
                "execution_time_ms": execution_time_ms,
                "success": success
            }
            
            # Add to session data
            session_data["tool_usage"].append(tool_entry)
            session_data["updated_at"] = tool_entry["timestamp"]
            
            # Save session file
            save_json_file(session_data, self.current_session_file)
            
            # Log to database
            self.db.log_tool_usage(
                session_id=session_data["id"],
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                execution_time_ms=execution_time_ms,
                success=success
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to log tool usage: {e}")
            return False
    
    def log_file_interaction(self, file_path: str, action_type: str,
                           line_changes: Dict = None) -> bool:
        """Log file interaction for current session."""
        try:
            session_data = self.get_current_session()
            if not session_data:
                logging.warning("No active session for file interaction logging")
                return False
            
            # Calculate file hash if file exists
            content_hash = calculate_file_hash(file_path)
            
            interaction_entry = {
                "file_path": str(file_path),
                "action_type": action_type,
                "timestamp": datetime.now().isoformat(),
                "content_hash": content_hash,
                "line_changes": line_changes
            }
            
            # Add to session data
            session_data["file_interactions"].append(interaction_entry)
            session_data["updated_at"] = interaction_entry["timestamp"]
            
            # Save session file
            save_json_file(session_data, self.current_session_file)
            
            # Log to database
            self.db.log_file_interaction(
                session_id=session_data["id"],
                file_path=file_path,
                action_type=action_type,
                content_hash=content_hash,
                line_changes=line_changes
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to log file interaction: {e}")
            return False
    
    def create_checkpoint(self, checkpoint_name: str = None) -> bool:
        """Create checkpoint of current session state."""
        try:
            session_data = self.get_current_session()
            if not session_data:
                logging.warning("No active session for checkpoint creation")
                return False
            
            if not checkpoint_name:
                checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint_data = {
                "name": checkpoint_name,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_data["id"],
                "git_info": get_git_info(str(self.project_root)),
                "session_state": session_data.copy()
            }
            
            # Save checkpoint
            checkpoint_file = (
                self.memory_dir / "sessions" / "checkpoints" / 
                f"{session_data['id']}_{checkpoint_name}.json"
            )
            save_json_file(checkpoint_data, checkpoint_file)
            
            # Update session with checkpoint reference
            session_data["checkpoints"].append({
                "name": checkpoint_name,
                "timestamp": checkpoint_data["timestamp"],
                "file": str(checkpoint_file)
            })
            save_json_file(session_data, self.current_session_file)
            
            logging.info(f"Created checkpoint: {checkpoint_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create checkpoint: {e}")
            return False
    
    def end_session(self, archive: bool = True) -> bool:
        """End current session and optionally archive it."""
        try:
            session_data = self.get_current_session()
            if not session_data:
                logging.warning("No active session to end")
                return False
            
            # Update session status
            session_data["status"] = "completed"
            session_data["ended_at"] = datetime.now().isoformat()
            
            # Update database
            self.db.update_session(session_data["id"], status="completed")
            
            if archive:
                # Move to archived sessions
                archive_file = (
                    self.memory_dir / "sessions" / "archived" / 
                    f"{session_data['id']}.json"
                )
                save_json_file(session_data, archive_file)
                
                # Remove current session file
                if self.current_session_file.exists():
                    self.current_session_file.unlink()
            else:
                # Just update current session file
                save_json_file(session_data, self.current_session_file)
            
            logging.info(f"Ended session: {session_data['id']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to end session: {e}")
            return False
    
    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get summary of session activity."""
        try:
            if session_id:
                session_data = self.db.get_session(session_id)
            else:
                session_data = self.get_current_session()
            
            if not session_data:
                return {}
            
            # Get tool usage from database
            tool_usage = self.db.get_session_tool_usage(session_data["id"])
            
            # Calculate summary statistics
            total_tools = len(tool_usage)
            tool_types = set(tool["tool_name"] for tool in tool_usage)
            
            # File interaction count
            file_count = len(session_data.get("file_interactions", []))
            
            # Duration calculation
            created_at = datetime.fromisoformat(session_data["created_at"])
            ended_at = session_data.get("ended_at")
            if ended_at:
                duration = datetime.fromisoformat(ended_at) - created_at
            else:
                duration = datetime.now() - created_at
            
            return {
                "session_id": session_data["id"],
                "created_at": session_data["created_at"],
                "duration_seconds": duration.total_seconds(),
                "status": session_data.get("status", "active"),
                "tool_usage_count": total_tools,
                "unique_tools_used": list(tool_types),
                "file_interaction_count": file_count,
                "checkpoints": len(session_data.get("checkpoints", [])),
                "git_info": session_data.get("git_info", {})
            }
            
        except Exception as e:
            logging.error(f"Failed to get session summary: {e}")
            return {}
    
    def cleanup_old_sessions(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up old sessions and files."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cleanup_stats = {
            "database_sessions_removed": 0,
            "archive_files_removed": 0,
            "checkpoint_files_removed": 0
        }
        
        try:
            # Clean database
            cleanup_stats["database_sessions_removed"] = self.db.cleanup_old_sessions(retention_days)
            
            # Clean archive files
            archive_dir = self.memory_dir / "sessions" / "archived"
            if archive_dir.exists():
                for file_path in archive_dir.glob("*.json"):
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                        file_path.unlink()
                        cleanup_stats["archive_files_removed"] += 1
            
            # Clean checkpoint files
            checkpoint_dir = self.memory_dir / "sessions" / "checkpoints"
            if checkpoint_dir.exists():
                for file_path in checkpoint_dir.glob("*.json"):
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                        file_path.unlink()
                        cleanup_stats["checkpoint_files_removed"] += 1
            
            logging.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logging.error(f"Failed to cleanup old sessions: {e}")
        
        return cleanup_stats