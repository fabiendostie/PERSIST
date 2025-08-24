#!/usr/bin/env python3
"""
Utility functions for Prsist Memory System.
Common helpers and validation functions.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(__file__).parent / "storage" / "memory.log"),
            logging.StreamHandler()
        ]
    )

def validate_path(path: str, allow_create: bool = False) -> bool:
    """Validate file/directory path to prevent traversal attacks."""
    try:
        resolved_path = Path(path).resolve()
        
        # Check for directory traversal
        if ".." in str(resolved_path):
            return False
        
        # Check if path exists or can be created
        if not resolved_path.exists() and allow_create:
            try:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
                return True
            except:
                return False
        
        return resolved_path.exists()
    except:
        return False

def sanitize_input(data: Any) -> Any:
    """Sanitize input data for database storage."""
    if isinstance(data, str):
        # Remove potentially dangerous characters
        return data.replace('\x00', '').strip()
    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    else:
        return data

def calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA-256 hash of file content."""
    try:
        if not validate_path(file_path):
            return None
        
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Failed to calculate hash for {file_path}: {e}")
        return None

def get_git_info(project_path: str = ".") -> Dict[str, str]:
    """Get current git information."""
    git_info = {}
    
    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["hash"] = result.stdout.strip()[:8]  # Short hash
        
        # Get status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["dirty"] = bool(result.stdout.strip())
        
    except Exception as e:
        logging.debug(f"Git info collection failed: {e}")
    
    return git_info

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        if not validate_path(config_path):
            return {}
        
        if not YAML_AVAILABLE:
            logging.warning("PyYAML not available, cannot load YAML config")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return {}

def save_yaml_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to YAML file."""
    try:
        if not YAML_AVAILABLE:
            logging.warning("PyYAML not available, cannot save YAML config")
            return False
            
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to save config to {config_path}: {e}")
        return False

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        if not validate_path(file_path):
            return {}
        
        # Check if file exists and is not empty
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {}
        
        if file_path_obj.stat().st_size == 0:
            logging.warning(f"JSON file is empty: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logging.warning(f"JSON file has no content: {file_path}")
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in {file_path}: {e}")
        # Backup corrupted file and return empty dict
        from datetime import datetime
        backup_path = f"{file_path}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            Path(file_path).rename(backup_path)
            logging.info(f"Corrupted JSON file backed up to: {backup_path}")
        except Exception as backup_e:
            logging.error(f"Failed to backup corrupted file: {backup_e}")
        return {}
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: str, 
                   indent: int = 2) -> bool:
    """Save data to JSON file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False

def get_project_root() -> Path:
    """Find project root by looking for key files."""
    current_path = Path.cwd()
    
    # Look for common project indicators
    indicators = [
        'package.json',
        'Cargo.toml',
        'pyproject.toml',
        'setup.py',
        '.git',
        'CLAUDE.md'
    ]
    
    for path in [current_path] + list(current_path.parents):
        if any((path / indicator).exists() for indicator in indicators):
            return path
    
    return current_path

def truncate_content(content: str, max_tokens: int = 50000) -> str:
    """Truncate content to approximate token limit."""
    # Rough approximation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(content) <= max_chars:
        return content
    
    # Try to truncate at word boundaries
    truncated = content[:max_chars]
    last_space = truncated.rfind(' ')
    
    if last_space > max_chars * 0.8:  # If we can find a space in the last 20%
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = ''.join(c if c not in unsafe_chars else '_' for c in name)
    
    # Limit length and strip
    return safe_name[:100].strip()

def get_memory_stats(memory_dir: str) -> Dict[str, Any]:
    """Get memory system statistics."""
    memory_path = Path(memory_dir)
    stats = {
        "total_sessions": 0,
        "active_session": None,
        "database_size_mb": 0,
        "last_activity": None
    }
    
    try:
        # Query database for session counts
        db_path = memory_path / "storage" / "sessions.db"
        if db_path.exists():
            import sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                # Get total session count
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats["total_sessions"] = cursor.fetchone()[0]
                
                # Get active session
                cursor = conn.execute("SELECT id FROM sessions WHERE status = 'active' ORDER BY created_at DESC LIMIT 1")
                active_row = cursor.fetchone()
                if active_row:
                    stats["active_session"] = active_row[0][:8] + "..."
                
                # Get database size
                stats["database_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)
                stats["last_activity"] = db_path.stat().st_mtime
    
    except Exception as e:
        logging.error(f"Failed to get memory stats: {e}")
    
    return stats