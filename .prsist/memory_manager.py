#!/usr/bin/env python3
"""
Core memory management module for Prsist Memory System.
Main interface for session management and memory operations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from database import MemoryDatabase
from session_tracker import SessionTracker
from context_builder import ContextBuilder
from enhanced_git_integration import EnhancedGitIntegrator
from utils import (
    setup_logging,
    load_yaml_config,
    get_memory_stats,
    get_project_root
)

class MemoryManager:
    """Main memory management class for Prsist Memory System."""
    
    def __init__(self, memory_dir: str = None, config_path: str = None):
        """Initialize memory manager."""
        # Set up paths
        if memory_dir is None:
            memory_dir = Path(__file__).parent
        
        self.memory_dir = Path(memory_dir)
        
        # Load configuration
        if config_path is None:
            yaml_config_path = self.memory_dir / "config" / "memory-config.yaml"
            json_config_path = self.memory_dir / "config" / "memory-config.json"
            
            # Try YAML first, fallback to JSON
            if yaml_config_path.exists():
                self.config = load_yaml_config(str(yaml_config_path))
            elif json_config_path.exists():
                from utils import load_json_file
                self.config = load_json_file(str(json_config_path))
            else:
                self.config = {}
        else:
            if str(config_path).endswith('.json'):
                from utils import load_json_file
                self.config = load_json_file(str(config_path))
            else:
                self.config = load_yaml_config(str(config_path))
        
        # Setup logging
        log_level = self.config.get("logging", {}).get("level", "INFO")
        setup_logging(log_level)
        
        # Initialize components
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        self.session_tracker = SessionTracker(str(self.memory_dir))
        self.context_builder = ContextBuilder(str(self.memory_dir), self.config)
        
        self.project_root = get_project_root()
        
        # Initialize enhanced git integration
        try:
            self.git_integrator = EnhancedGitIntegrator(str(self.memory_dir), str(self.project_root))
            self.git_integration_enabled = True
            logging.info("Enhanced Git Integration initialized")
        except Exception as e:
            logging.warning(f"Git integration disabled: {e}")
            self.git_integrator = None
            self.git_integration_enabled = False
        
        logging.info("Memory manager initialized")
    
    def start_session(self, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start new memory session."""
        try:
            logging.info("Starting new memory session")
            
            # Build initial context
            if self.context_builder.auto_inject:
                initial_context = self.context_builder.build_session_context()
                if context_data:
                    initial_context.update(context_data)
                context_data = initial_context
            
            # Start session tracking
            session_result = self.session_tracker.start_session(context_data)
            
            if session_result.get("memory_system_active"):
                session_id = session_result['session_id']
                logging.info(f"Memory session started: {session_id}")
                
                # Auto-correlate with git if enabled
                if self.git_integration_enabled and self.git_integrator:
                    try:
                        git_correlation = self.git_integrator.auto_correlate_session(session_id)
                        session_result['git_correlation'] = git_correlation
                        if git_correlation.get('correlated'):
                            logging.info(f"Session {session_id} auto-correlated with git")
                    except Exception as e:
                        logging.warning(f"Git auto-correlation failed: {e}")
            else:
                logging.error("Failed to start memory session")
            
            return session_result
            
        except Exception as e:
            logging.error(f"Failed to start memory session: {e}")
            return {
                "session_id": None,
                "memory_system_active": False,
                "error": str(e)
            }
    
    def get_session_context(self) -> str:
        """Get formatted context for Claude Code."""
        try:
            return self.context_builder.get_context_for_claude()
        except Exception as e:
            logging.error(f"Failed to get session context: {e}")
            return ""
    
    def log_tool_usage(self, tool_name: str, input_data: Any = None,
                      output_data: Any = None, execution_time_ms: int = None,
                      success: bool = True) -> bool:
        """Log tool usage for current session."""
        try:
            return self.session_tracker.log_tool_usage(
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                execution_time_ms=execution_time_ms,
                success=success
            )
        except Exception as e:
            logging.error(f"Failed to log tool usage: {e}")
            return False
    
    def log_file_interaction(self, file_path: str, action_type: str,
                           line_changes: Dict = None) -> bool:
        """Log file interaction for current session."""
        try:
            return self.session_tracker.log_file_interaction(
                file_path=file_path,
                action_type=action_type,
                line_changes=line_changes
            )
        except Exception as e:
            logging.error(f"Failed to log file interaction: {e}")
            return False
    
    def update_session_context(self, context_updates: Dict[str, Any]) -> bool:
        """Update current session context."""
        try:
            return self.session_tracker.update_session(context_data=context_updates)
        except Exception as e:
            logging.error(f"Failed to update session context: {e}")
            return False
    
    def create_checkpoint(self, checkpoint_name: str = None) -> bool:
        """Create checkpoint of current session."""
        try:
            return self.session_tracker.create_checkpoint(checkpoint_name)
        except Exception as e:
            logging.error(f"Failed to create checkpoint: {e}")
            return False
    
    def end_session(self, archive: bool = True) -> bool:
        """End current session."""
        try:
            result = self.session_tracker.end_session(archive)
            if result:
                logging.info("Memory session ended successfully")
            return result
        except Exception as e:
            logging.error(f"Failed to end session: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        try:
            current_session = self.session_tracker.get_current_session()
            if current_session:
                return self.session_tracker.get_session_summary()
            else:
                return {"status": "no_active_session"}
        except Exception as e:
            logging.error(f"Failed to get session info: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            stats = get_memory_stats(str(self.memory_dir))
            stats["config"] = self.config
            stats["project_root"] = str(self.project_root)
            return stats
        except Exception as e:
            logging.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def add_project_memory(self, content: str, append: bool = True) -> bool:
        """Add content to project memory."""
        try:
            return self.context_builder.update_project_memory(content, append)
        except Exception as e:
            logging.error(f"Failed to add project memory: {e}")
            return False
    
    def add_decision(self, title: str, description: str,
                    category: str = "general", impact: str = "medium") -> bool:
        """Add a decision to project decisions log."""
        try:
            return self.context_builder.add_decision(title, description, category, impact)
        except Exception as e:
            logging.error(f"Failed to add decision: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: int = None) -> Dict[str, Any]:
        """Clean up old memory data."""
        try:
            if retention_days is None:
                retention_days = self.config.get("storage", {}).get("retention_days", 30)
            
            cleanup_stats = self.session_tracker.cleanup_old_sessions(retention_days)
            logging.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return {"error": str(e)}
    
    def validate_system(self) -> Dict[str, Any]:
        """Validate memory system integrity."""
        validation_results = {
            "valid": True,
            "issues": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check directory structure
            required_dirs = [
                "config",
                "sessions/active",
                "sessions/archived", 
                "sessions/checkpoints",
                "context",
                "storage"
            ]
            
            for dir_path in required_dirs:
                full_path = self.memory_dir / dir_path
                if not full_path.exists():
                    validation_results["issues"].append(f"Missing directory: {dir_path}")
                    validation_results["valid"] = False
            
            # Check database connectivity
            try:
                self.db.get_recent_sessions(1)
            except Exception as e:
                validation_results["issues"].append(f"Database connection failed: {e}")
                validation_results["valid"] = False
            
            # Check configuration
            required_config_keys = ["memory_system", "storage", "session", "context"]
            for key in required_config_keys:
                if key not in self.config:
                    validation_results["issues"].append(f"Missing config section: {key}")
                    validation_results["valid"] = False
            
            # Check file permissions
            test_file = self.memory_dir / "storage" / "test_write.tmp"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                validation_results["issues"].append(f"Write permission test failed: {e}")
                validation_results["valid"] = False
            
            if validation_results["valid"]:
                logging.info("Memory system validation passed")
            else:
                logging.warning(f"Memory system validation failed: {validation_results['issues']}")
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Validation error: {e}")
            logging.error(f"Memory system validation error: {e}")
        
        return validation_results
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session summaries."""
        try:
            sessions = self.db.get_recent_sessions(limit)
            summaries = []
            for session in sessions:
                summary = self.session_tracker.get_session_summary(session["id"])
                if summary:
                    summaries.append(summary)
            return summaries
        except Exception as e:
            logging.error(f"Failed to get recent sessions: {e}")
            return []
    
    def export_session_data(self, session_id: str = None, 
                           format: str = "json") -> Optional[str]:
        """Export session data for analysis or backup."""
        try:
            if session_id:
                session_data = self.db.get_session(session_id)
                tool_usage = self.db.get_session_tool_usage(session_id)
            else:
                session_data = self.session_tracker.get_current_session()
                if session_data:
                    tool_usage = self.db.get_session_tool_usage(session_data["id"])
                else:
                    return None
            
            if not session_data:
                return None
            
            export_data = {
                "session": session_data,
                "tool_usage": tool_usage,
                "export_timestamp": datetime.now().isoformat(),
                "memory_system_version": "1.0"
            }
            
            if format.lower() == "json":
                import json
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                return str(export_data)
                
        except Exception as e:
            logging.error(f"Failed to export session data: {e}")
            return None
    
    def cli_bridge_handler(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Handle commands from JavaScript bridge."""
        args = args or []
        
        try:
            if command == 'start_session':
                metadata = {}
                if args and args[0]:
                    import json
                    metadata = json.loads(args[0])
                session = self.start_session(**metadata)
                return {"status": "success", "data": session}
            
            elif command == 'end_session':
                session_id = args[0] if args else None
                result = self.end_session(session_id)
                return {"status": "success", "data": result}
            
            elif command == 'create_checkpoint':
                name = args[0] if args else "checkpoint"
                description = args[1] if len(args) > 1 else ""
                result = self.create_checkpoint(name, description)
                return {"status": "success", "data": result}
            
            elif command == 'get_context':
                include_decisions = args[0].lower() == 'true' if args else True
                context = self.get_session_context()
                return {"status": "success", "data": {"content": context}}
            
            elif command == 'add_memory':
                content = args[0] if args else ""
                memory_type = args[1] if len(args) > 1 else "note"
                result = self.add_project_memory(content, memory_type)
                return {"status": "success", "data": result}
            
            elif command == 'add_decision':
                decision = args[0] if args else ""
                rationale = args[1] if len(args) > 1 else ""
                impact = args[2] if len(args) > 2 else "medium"
                result = self.add_decision(decision, rationale, impact)
                return {"status": "success", "data": result}
            
            elif command == 'capture_event':
                if args and args[0]:
                    import json
                    event_data = json.loads(args[0])
                    result = self.capture_workflow_event(event_data)
                    return {"status": "success", "data": result}
                return {"status": "error", "message": "No event data provided"}
            
            elif command == 'correlate_git':
                if args and args[0]:
                    import json
                    git_data = json.loads(args[0])
                    result = self.correlate_with_git(git_data)
                    return {"status": "success", "data": result}
                return {"status": "error", "message": "No git data provided"}
            
            elif command == 'get_stats':
                stats = get_memory_stats(self.memory_dir)
                return {"status": "success", "data": stats}
            
            elif command == 'get_recent_sessions':
                limit = int(args[0]) if args else 10
                sessions = self.get_recent_sessions(limit)
                return {"status": "success", "data": sessions}
            
            elif command == 'health_check':
                health = self.health_check()
                return {"status": "success", "data": health}
            
            elif command == 'validate':
                validation = self.validate_system()
                return {"status": "success", "data": validation}
            
            elif command == 'git_switch_branch':
                if len(args) >= 3:
                    from_branch, to_branch, session_id = args[0], args[1], args[2]
                    if self.git_integration_enabled:
                        result = self.git_integrator.switch_branch_context(from_branch, to_branch, session_id)
                        return {"status": "success", "data": result}
                    else:
                        return {"status": "error", "message": "Git integration not available"}
                return {"status": "error", "message": "from_branch, to_branch, and session_id required"}
            
            elif command == 'git_check_updates':
                if self.git_integration_enabled:
                    result = self.git_integrator.check_for_correlation_updates()
                    return {"status": "success", "data": result}
                else:
                    return {"status": "error", "message": "Git integration not available"}
            
            elif command == 'git_report':
                session_id = args[0] if args else None
                branch_name = args[1] if len(args) > 1 else None
                if self.git_integration_enabled:
                    result = self.git_integrator.generate_git_memory_report(session_id, branch_name)
                    return {"status": "success", "data": result}
                else:
                    return {"status": "error", "message": "Git integration not available"}
            
            elif command == 'git_track_merge':
                if args and len(args) >= 2:
                    merge_commit_sha, session_id = args[0], args[1]
                    if self.git_integration_enabled:
                        result = self.git_integrator.track_merge_operation(merge_commit_sha, session_id)
                        return {"status": "success", "data": result}
                    else:
                        return {"status": "error", "message": "Git integration not available"}
                return {"status": "error", "message": "merge_commit_sha and session_id required"}
            
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
                
        except Exception as e:
            logging.error(f"CLI bridge error for command '{command}': {e}")
            return {"status": "error", "message": str(e)}
    
    def capture_workflow_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Capture workflow-specific events (BMAD, etc.)."""
        try:
            # Store workflow event in database
            event_id = self.db.store_workflow_event(
                event_type=event_data.get('type', 'unknown'),
                workflow=event_data.get('workflow', 'generic'),
                event_data=event_data.get('data', {}),
                timestamp=event_data.get('timestamp', datetime.now().isoformat())
            )
            
            # Update project memory if significant
            if event_data.get('type') in ['agent_decision', 'architecture_decision']:
                memory_content = self._format_workflow_event_for_memory(event_data)
                self.add_project_memory(memory_content, 'workflow_event')
            
            return {"event_id": event_id, "captured": True}
            
        except Exception as e:
            logging.error(f"Failed to capture workflow event: {e}")
            return {"error": str(e), "captured": False}
    
    def correlate_with_git(self, git_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate current session with git information."""
        try:
            current_session = self.session_tracker.get_current_session()
            if not current_session:
                return {"error": "No active session", "correlated": False}
            
            # Store git correlation
            correlation_id = self.db.store_git_correlation(
                session_id=current_session["id"],
                commit_hash=git_data.get('commit'),
                branch_name=git_data.get('branch'),
                timestamp=git_data.get('timestamp', datetime.now().isoformat())
            )
            
            return {"correlation_id": correlation_id, "correlated": True}
            
        except Exception as e:
            logging.error(f"Failed to correlate with git: {e}")
            return {"error": str(e), "correlated": False}
    
    def _format_workflow_event_for_memory(self, event_data: Dict[str, Any]) -> str:
        """Format workflow event for project memory."""
        event_type = event_data.get('type', 'unknown')
        workflow = event_data.get('workflow', 'generic')
        data = event_data.get('data', {})
        
        if event_type == 'agent_decision':
            agent = data.get('agent', 'unknown')
            decision = data.get('decision', 'unknown')
            return f"[{workflow.upper()}] {agent} decided: {decision}"
        
        elif event_type == 'architecture_decision':
            component = data.get('component', 'unknown')
            decision = data.get('decision', 'unknown')
            return f"[ARCHITECTURE] {component}: {decision}"
        
        elif event_type == 'story_event':
            title = data.get('story_title', 'unknown')
            event_type = data.get('event_type', 'unknown')
            return f"[STORY] {title} - {event_type}"
        
        else:
            return f"[{workflow.upper()}] {event_type}: {data}"


# CLI interface for JavaScript bridge
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Initialize memory manager
    manager = MemoryManager()
    
    # Handle command
    result = manager.cli_bridge_handler(command, args)
    
    # Output JSON result
    print(json.dumps(result, ensure_ascii=False, indent=None))