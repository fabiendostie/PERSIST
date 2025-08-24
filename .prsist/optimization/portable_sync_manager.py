#!/usr/bin/env python3
"""
Portable sync mechanisms for Prsist Memory System Phase 4.
Implements cross-machine synchronization with multiple backends.
"""

import os
import subprocess
import json
import shutil
import hashlib
import logging
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from utils import setup_logging

@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    status: str  # success, error, conflict
    conflicts: List[Dict[str, Any]]
    synced_items: List[str]
    errors: List[str]
    pulled: Optional[Dict[str, Any]] = None
    pushed: Optional[Dict[str, Any]] = None
    sync_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.sync_timestamp:
            result['sync_timestamp'] = self.sync_timestamp.isoformat()
        return result

@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""
    conflict_id: str
    conflict_type: str  # content_modification, deletion_modification, concurrent_addition
    local_content: Any
    remote_content: Any
    file_path: str
    resolution_strategy: Optional[str] = None
    merged_content: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class ConflictResolver:
    """Resolves synchronization conflicts."""
    
    def __init__(self):
        self.resolution_strategies = {
            'auto_merge': self._auto_merge_strategy,
            'use_newer': self._use_newer_strategy,
            'use_local': self._use_local_strategy,
            'use_remote': self._use_remote_strategy,
            'keep_modified': self._keep_modified_strategy,
            'merge_both': self._merge_both_strategy
        }
    
    def resolve_conflicts(self, conflicts: List[SyncConflict], 
                         default_strategy: str = 'auto_merge') -> List[Dict[str, Any]]:
        """Resolve a list of conflicts."""
        try:
            resolved_conflicts = []
            
            for conflict in conflicts:
                resolution = self._resolve_single_conflict(conflict, default_strategy)
                resolved_conflicts.append(resolution)
            
            return resolved_conflicts
            
        except Exception as e:
            logging.error(f"Failed to resolve conflicts: {e}")
            return []
    
    def _resolve_single_conflict(self, conflict: SyncConflict, 
                               default_strategy: str) -> Dict[str, Any]:
        """Resolve a single conflict."""
        try:
            # Determine resolution strategy
            strategy = self._determine_strategy(conflict, default_strategy)
            
            # Apply resolution
            if strategy in self.resolution_strategies:
                merged_content = self.resolution_strategies[strategy](conflict)
            else:
                merged_content = self._auto_merge_strategy(conflict)
                strategy = 'auto_merge'
            
            return {
                'conflict_id': conflict.conflict_id,
                'resolution_strategy': strategy,
                'merged_content': merged_content,
                'file_path': conflict.file_path,
                'conflict_type': conflict.conflict_type
            }
            
        except Exception as e:
            logging.error(f"Failed to resolve conflict {conflict.conflict_id}: {e}")
            return {
                'conflict_id': conflict.conflict_id,
                'resolution_strategy': 'error',
                'merged_content': conflict.local_content,
                'error': str(e)
            }
    
    def _determine_strategy(self, conflict: SyncConflict, default: str) -> str:
        """Determine the best resolution strategy for a conflict."""
        try:
            if conflict.conflict_type == 'deletion_modification':
                return 'keep_modified'
            elif conflict.conflict_type == 'concurrent_addition':
                return 'merge_both'
            elif conflict.conflict_type == 'content_modification':
                if self._can_auto_merge(conflict):
                    return 'auto_merge'
                else:
                    return 'use_newer'
            else:
                return default
                
        except Exception as e:
            logging.error(f"Failed to determine strategy: {e}")
            return default
    
    def _can_auto_merge(self, conflict: SyncConflict) -> bool:
        """Check if conflict can be auto-merged."""
        try:
            # Simple heuristic: can merge if changes are in different parts
            local_str = str(conflict.local_content)
            remote_str = str(conflict.remote_content)
            
            # If content types are different, can't merge
            if type(conflict.local_content) != type(conflict.remote_content):
                return False
            
            # For small content, don't auto-merge
            if len(local_str) < 100 or len(remote_str) < 100:
                return False
            
            # Simple check: if strings have common prefix/suffix, might be mergeable
            if isinstance(conflict.local_content, str):
                common_prefix_len = 0
                min_len = min(len(local_str), len(remote_str))
                
                for i in range(min_len):
                    if local_str[i] == remote_str[i]:
                        common_prefix_len += 1
                    else:
                        break
                
                # If more than 50% is common, consider it mergeable
                return common_prefix_len > min_len * 0.5
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to check if can auto-merge: {e}")
            return False
    
    def _auto_merge_strategy(self, conflict: SyncConflict) -> Any:
        """Auto-merge strategy implementation."""
        try:
            if isinstance(conflict.local_content, str) and isinstance(conflict.remote_content, str):
                # Simple line-based merge for text content
                local_lines = conflict.local_content.splitlines()
                remote_lines = conflict.remote_content.splitlines()
                
                # Find common base and merge
                merged_lines = self._merge_text_lines(local_lines, remote_lines)
                return '\n'.join(merged_lines)
            
            elif isinstance(conflict.local_content, dict) and isinstance(conflict.remote_content, dict):
                # Merge dictionaries
                merged = conflict.local_content.copy()
                merged.update(conflict.remote_content)
                return merged
            
            elif isinstance(conflict.local_content, list) and isinstance(conflict.remote_content, list):
                # Merge lists (combine and deduplicate)
                merged = list(conflict.local_content)
                for item in conflict.remote_content:
                    if item not in merged:
                        merged.append(item)
                return merged
            
            else:
                # Fallback to newer version
                return self._use_newer_strategy(conflict)
                
        except Exception as e:
            logging.error(f"Failed to auto-merge: {e}")
            return conflict.local_content
    
    def _merge_text_lines(self, local_lines: List[str], remote_lines: List[str]) -> List[str]:
        """Merge text lines using simple strategy."""
        try:
            # Very simple merge: combine unique lines
            merged = []
            all_lines = local_lines + remote_lines
            seen = set()
            
            for line in all_lines:
                if line not in seen:
                    merged.append(line)
                    seen.add(line)
            
            return merged
            
        except Exception as e:
            logging.error(f"Failed to merge text lines: {e}")
            return local_lines
    
    def _use_newer_strategy(self, conflict: SyncConflict) -> Any:
        """Use newer version strategy."""
        # For this implementation, assume remote is newer
        return conflict.remote_content
    
    def _use_local_strategy(self, conflict: SyncConflict) -> Any:
        """Use local version strategy."""
        return conflict.local_content
    
    def _use_remote_strategy(self, conflict: SyncConflict) -> Any:
        """Use remote version strategy."""
        return conflict.remote_content
    
    def _keep_modified_strategy(self, conflict: SyncConflict) -> Any:
        """Keep modified version strategy."""
        # If one side deleted and other modified, keep modified
        if conflict.local_content is None:
            return conflict.remote_content
        elif conflict.remote_content is None:
            return conflict.local_content
        else:
            return self._use_newer_strategy(conflict)
    
    def _merge_both_strategy(self, conflict: SyncConflict) -> Any:
        """Merge both versions strategy."""
        return self._auto_merge_strategy(conflict)

class GitSyncStrategy:
    """Git-based synchronization strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_path = Path(config.get('repository_path', '.prsist-sync'))
        self.remote_url = config.get('remote_url', '')
        self.branch = config.get('branch', 'memory-sync')
        
    def initialize_repository(self) -> bool:
        """Initialize git repository for sync."""
        try:
            if not self.repo_path.exists():
                self.repo_path.mkdir(parents=True)
                
                # Initialize git repo
                result = subprocess.run(
                    ['git', 'init'],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logging.error(f"Failed to initialize git repo: {result.stderr}")
                    return False
                
                # Add remote if provided
                if self.remote_url:
                    result = subprocess.run(
                        ['git', 'remote', 'add', 'origin', self.remote_url],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logging.warning(f"Failed to add remote: {result.stderr}")
                
                # Create initial structure
                self._create_sync_structure()
                
                # Initial commit
                self._commit_changes("Initial memory sync setup")
                
                logging.info(f"Initialized git repository at {self.repo_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize repository: {e}")
            return False
    
    def _create_sync_structure(self):
        """Create initial sync directory structure."""
        try:
            (self.repo_path / 'sessions').mkdir(exist_ok=True)
            (self.repo_path / 'cache').mkdir(exist_ok=True)
            (self.repo_path / 'config').mkdir(exist_ok=True)
            (self.repo_path / 'metadata').mkdir(exist_ok=True)
            
            # Create README
            readme_content = """# BMAD Memory Sync Repository

This repository contains synchronized memory data for the Prsist System memory system.

## Structure
- sessions/: Session data and context
- cache/: Cached prefixes and optimization data
- config/: Configuration files
- metadata/: Sync metadata and statistics
"""
            
            with open(self.repo_path / 'README.md', 'w') as f:
                f.write(readme_content)
                
        except Exception as e:
            logging.error(f"Failed to create sync structure: {e}")
    
    def pull_changes(self) -> Dict[str, Any]:
        """Pull changes from remote repository."""
        try:
            if not self.remote_url:
                return {'status': 'skipped', 'reason': 'No remote URL configured'}
            
            # Fetch changes
            result = subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'status': 'error', 'error': result.stderr}
            
            # Check for conflicts
            result = subprocess.run(
                ['git', 'merge', f'origin/{self.branch}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Handle merge conflicts
                conflicts = self._detect_conflicts()
                return {
                    'status': 'conflict',
                    'conflicts': conflicts,
                    'output': result.stderr
                }
            
            return {
                'status': 'success',
                'changes_pulled': self._count_changes(result.stdout)
            }
            
        except Exception as e:
            logging.error(f"Failed to pull changes: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def push_changes(self) -> Dict[str, Any]:
        """Push changes to remote repository."""
        try:
            if not self.remote_url:
                return {'status': 'skipped', 'reason': 'No remote URL configured'}
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path)
            
            # Check if there are changes to commit
            result = subprocess.run(
                ['git', 'diff', '--cached', '--quiet'],
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                return {'status': 'skipped', 'reason': 'No changes to push'}
            
            # Commit changes
            commit_message = self._generate_commit_message()
            result = subprocess.run(
                ['git', 'commit', '-m', commit_message],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'status': 'error', 'error': result.stderr}
            
            # Push to remote
            result = subprocess.run(
                ['git', 'push', 'origin', self.branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'status': 'error', 'error': result.stderr}
            
            return {
                'status': 'success',
                'commit_hash': self._get_last_commit_hash()
            }
            
        except Exception as e:
            logging.error(f"Failed to push changes: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _commit_changes(self, message: str) -> bool:
        """Commit changes to repository."""
        try:
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path)
            
            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Failed to commit changes: {e}")
            return False
    
    def _generate_commit_message(self) -> str:
        """Generate commit message."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        template = self.config.get('commit_message_template', 'Memory sync: {timestamp}')
        return template.format(timestamp=timestamp)
    
    def _get_last_commit_hash(self) -> str:
        """Get last commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            return ''
            
        except Exception as e:
            logging.error(f"Failed to get commit hash: {e}")
            return ''
    
    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect merge conflicts."""
        try:
            # Get list of conflicted files
            result = subprocess.run(
                ['git', 'diff', '--name-only', '--diff-filter=U'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            conflicts = []
            if result.returncode == 0 and result.stdout:
                for file_path in result.stdout.strip().split('\n'):
                    if file_path:
                        conflicts.append({
                            'file_path': file_path,
                            'type': 'merge_conflict'
                        })
            
            return conflicts
            
        except Exception as e:
            logging.error(f"Failed to detect conflicts: {e}")
            return []
    
    def _count_changes(self, output: str) -> int:
        """Count number of changes from git output."""
        try:
            # Simple heuristic: count lines mentioning files
            lines = output.split('\n')
            return len([line for line in lines if 'file' in line.lower()])
        except Exception:
            return 0

class RsyncStrategy:
    """Rsync-based synchronization strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_path = Path(config.get('local_path', '.prsist'))
        self.remote_path = config.get('remote_path', '')
        
    def sync_to_remote(self) -> Dict[str, Any]:
        """Sync local files to remote."""
        try:
            if not self.remote_path:
                return {'status': 'error', 'error': 'No remote path configured'}
            
            result = subprocess.run([
                'rsync', '-avz', '--delete',
                str(self.local_path) + '/',
                self.remote_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {'status': 'success', 'output': result.stdout}
            else:
                return {'status': 'error', 'error': result.stderr}
                
        except Exception as e:
            logging.error(f"Failed to sync to remote: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def sync_from_remote(self) -> Dict[str, Any]:
        """Sync files from remote to local."""
        try:
            if not self.remote_path:
                return {'status': 'error', 'error': 'No remote path configured'}
            
            result = subprocess.run([
                'rsync', '-avz', '--delete',
                self.remote_path + '/',
                str(self.local_path) + '/'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return {'status': 'success', 'output': result.stdout}
            else:
                return {'status': 'error', 'error': result.stderr}
                
        except Exception as e:
            logging.error(f"Failed to sync from remote: {e}")
            return {'status': 'error', 'error': str(e)}

class CloudSyncStrategy:
    """Cloud-based synchronization strategy (placeholder)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'generic')
        
    def sync_to_cloud(self) -> Dict[str, Any]:
        """Sync to cloud storage."""
        return {'status': 'not_implemented', 'message': 'Cloud sync not yet implemented'}
    
    def sync_from_cloud(self) -> Dict[str, Any]:
        """Sync from cloud storage."""
        return {'status': 'not_implemented', 'message': 'Cloud sync not yet implemented'}

class CrossPlatformSync:
    """Cross-platform synchronization utilities."""
    
    def __init__(self):
        self.platform = self._detect_platform()
        
    def _detect_platform(self) -> str:
        """Detect current platform."""
        import platform
        system = platform.system().lower()
        if system == 'windows':
            return 'windows'
        elif system == 'darwin':
            return 'macos'
        else:
            return 'linux'
    
    def setup_xdg_structure(self) -> Dict[str, Path]:
        """Setup XDG Base Directory structure for cross-platform compatibility."""
        try:
            xdg_paths = {
                'config': self.get_xdg_config_home() / 'prsist',
                'data': self.get_xdg_data_home() / 'prsist',
                'cache': self.get_xdg_cache_home() / 'prsist',
                'state': self.get_xdg_state_home() / 'prsist'
            }
            
            # Create directories if they don't exist
            for path_type, path in xdg_paths.items():
                path.mkdir(parents=True, exist_ok=True)
                
                # Create structure based on type
                if path_type == 'config':
                    self._create_config_structure(path)
                elif path_type == 'data':
                    self._create_data_structure(path)
                elif path_type == 'cache':
                    self._create_cache_structure(path)
                elif path_type == 'state':
                    self._create_state_structure(path)
            
            return xdg_paths
            
        except Exception as e:
            logging.error(f"Failed to setup XDG structure: {e}")
            return {}
    
    def get_xdg_config_home(self) -> Path:
        """Get XDG config directory (cross-platform)."""
        if self.platform == 'windows':
            appdata = os.environ.get('APPDATA', '')
            return Path(appdata) if appdata else Path.home() / 'AppData' / 'Roaming'
        else:
            xdg_config = os.environ.get('XDG_CONFIG_HOME', '')
            return Path(xdg_config) if xdg_config else Path.home() / '.config'
    
    def get_xdg_data_home(self) -> Path:
        """Get XDG data directory (cross-platform)."""
        if self.platform == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA', '')
            return Path(localappdata) if localappdata else Path.home() / 'AppData' / 'Local'
        else:
            xdg_data = os.environ.get('XDG_DATA_HOME', '')
            return Path(xdg_data) if xdg_data else Path.home() / '.local' / 'share'
    
    def get_xdg_cache_home(self) -> Path:
        """Get XDG cache directory (cross-platform)."""
        if self.platform == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA', '')
            cache_path = Path(localappdata) / 'cache' if localappdata else Path.home() / 'AppData' / 'Local' / 'cache'
            return cache_path
        else:
            xdg_cache = os.environ.get('XDG_CACHE_HOME', '')
            return Path(xdg_cache) if xdg_cache else Path.home() / '.cache'
    
    def get_xdg_state_home(self) -> Path:
        """Get XDG state directory (cross-platform)."""
        if self.platform == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA', '')
            return Path(localappdata) if localappdata else Path.home() / 'AppData' / 'Local'
        else:
            xdg_state = os.environ.get('XDG_STATE_HOME', '')
            return Path(xdg_state) if xdg_state else Path.home() / '.local' / 'state'
    
    def _create_config_structure(self, path: Path):
        """Create configuration structure."""
        (path / 'sync').mkdir(exist_ok=True)
        (path / 'profiles').mkdir(exist_ok=True)
    
    def _create_data_structure(self, path: Path):
        """Create data structure."""
        (path / 'sessions').mkdir(exist_ok=True)
        (path / 'knowledge').mkdir(exist_ok=True)
    
    def _create_cache_structure(self, path: Path):
        """Create cache structure."""
        (path / 'prefixes').mkdir(exist_ok=True)
        (path / 'embeddings').mkdir(exist_ok=True)
    
    def _create_state_structure(self, path: Path):
        """Create state structure."""
        (path / 'sync').mkdir(exist_ok=True)
        (path / 'locks').mkdir(exist_ok=True)

class PortableSyncManager:
    """Main portable synchronization manager."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.sync_backend = self._initialize_backend()
        self.conflict_resolver = ConflictResolver()
        self.cross_platform = CrossPlatformSync()
        
        # Setup XDG structure
        self.xdg_paths = self.cross_platform.setup_xdg_structure()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load sync configuration."""
        try:
            if not self.config_path.exists():
                return self._create_default_config()
            
            if YAML_AVAILABLE and self.config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            'sync': {
                'enabled': True,
                'strategy': 'git',
                'conflict_resolution': 'auto',
                'sync_interval': 300,
                'compression': True,
                'incremental': True,
                'git_sync': {
                    'repository_path': '.prsist-sync',
                    'remote_url': '',
                    'branch': 'memory-sync',
                    'auto_commit': True,
                    'commit_message_template': 'Memory sync: {timestamp}'
                },
                'cross_platform': {
                    'xdg_compliance': True,
                    'path_normalization': True,
                    'encoding_standardization': True
                },
                'conflict_resolution_strategies': {
                    'content_modification': 'auto_merge',
                    'deletion_modification': 'keep_modified',
                    'concurrent_additions': 'merge_both'
                }
            }
        }
    
    def _initialize_backend(self) -> Any:
        """Initialize sync backend."""
        try:
            strategy = self.config.get('sync', {}).get('strategy', 'git')
            
            if strategy == 'git':
                git_config = self.config.get('sync', {}).get('git_sync', {})
                return GitSyncStrategy(git_config)
            elif strategy == 'rsync':
                rsync_config = self.config.get('sync', {}).get('rsync_sync', {})
                return RsyncStrategy(rsync_config)
            elif strategy == 'cloud':
                cloud_config = self.config.get('sync', {}).get('cloud_sync', {})
                return CloudSyncStrategy(cloud_config)
            else:
                logging.warning(f"Unknown sync strategy: {strategy}, defaulting to git")
                git_config = self.config.get('sync', {}).get('git_sync', {})
                return GitSyncStrategy(git_config)
                
        except Exception as e:
            logging.error(f"Failed to initialize backend: {e}")
            return GitSyncStrategy({})
    
    def setup_git_based_sync(self) -> bool:
        """Setup git-based synchronization for memory data."""
        try:
            if isinstance(self.sync_backend, GitSyncStrategy):
                return self.sync_backend.initialize_repository()
            else:
                logging.error("Cannot setup git sync with non-git backend")
                return False
                
        except Exception as e:
            logging.error(f"Failed to setup git-based sync: {e}")
            return False
    
    def sync_memory_state(self, direction: str = 'both') -> SyncResult:
        """Synchronize memory state across machines."""
        try:
            conflicts = []
            synced_items = []
            errors = []
            pulled = None
            pushed = None
            
            if direction in ['pull', 'both']:
                # Pull remote changes
                if hasattr(self.sync_backend, 'pull_changes'):
                    pulled = self.sync_backend.pull_changes()
                    if pulled.get('status') == 'conflict':
                        conflicts.extend(pulled.get('conflicts', []))
                    elif pulled.get('status') == 'error':
                        errors.append(pulled.get('error', 'Unknown pull error'))
                    else:
                        synced_items.extend(['pulled_changes'])
            
            if direction in ['push', 'both']:
                # Push local changes
                if hasattr(self.sync_backend, 'push_changes'):
                    pushed = self.sync_backend.push_changes()
                    if pushed.get('status') == 'error':
                        errors.append(pushed.get('error', 'Unknown push error'))
                    else:
                        synced_items.extend(['pushed_changes'])
            
            # Resolve conflicts if any
            if conflicts:
                default_strategy = self.config.get('sync', {}).get('conflict_resolution', 'auto')
                resolved = self.conflict_resolver.resolve_conflicts(conflicts, default_strategy)
                # Apply resolved conflicts
                self._apply_conflict_resolutions(resolved)
            
            # Determine overall status
            if errors:
                status = 'error'
            elif conflicts:
                status = 'conflict'
            else:
                status = 'success'
            
            return SyncResult(
                status=status,
                conflicts=conflicts,
                synced_items=synced_items,
                errors=errors,
                pulled=pulled,
                pushed=pushed,
                sync_timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Failed to sync memory state: {e}")
            return SyncResult(
                status='error',
                conflicts=[],
                synced_items=[],
                errors=[str(e)],
                sync_timestamp=datetime.now()
            )
    
    def implement_incremental_sync(self) -> Dict[str, Any]:
        """Implement incremental synchronization for efficiency."""
        try:
            # Get last sync timestamp
            last_sync = self._get_last_sync_timestamp()
            
            # Find changes since last sync
            local_changes = self._find_local_changes_since(last_sync)
            
            # Create sync package
            sync_package = {
                'machine_id': self._get_machine_id(),
                'timestamp': datetime.now().isoformat(),
                'changes': []
            }
            
            for change in local_changes:
                change_entry = {
                    'type': change['type'],  # 'create', 'update', 'delete'
                    'path': change['path'],
                    'content': change.get('content'),
                    'hash': self._calculate_hash(change.get('content', '')),
                    'metadata': change.get('metadata', {})
                }
                sync_package['changes'].append(change_entry)
            
            # Apply compression if package is large
            if self._get_package_size(sync_package) > 1024 * 1024:  # 1MB
                sync_package = self._compress_sync_package(sync_package)
            
            # Update last sync timestamp
            self._update_last_sync_timestamp(datetime.now())
            
            return sync_package
            
        except Exception as e:
            logging.error(f"Failed to implement incremental sync: {e}")
            return {'error': str(e)}
    
    def _apply_conflict_resolutions(self, resolved_conflicts: List[Dict[str, Any]]):
        """Apply resolved conflicts."""
        try:
            for resolution in resolved_conflicts:
                file_path = resolution.get('file_path', '')
                merged_content = resolution.get('merged_content')
                
                if file_path and merged_content is not None:
                    # Write merged content to file
                    full_path = Path(file_path)
                    if full_path.parent.exists():
                        if isinstance(merged_content, str):
                            with open(full_path, 'w') as f:
                                f.write(merged_content)
                        else:
                            with open(full_path, 'w') as f:
                                json.dump(merged_content, f, indent=2)
                        
                        logging.info(f"Applied conflict resolution for {file_path}")
                    
        except Exception as e:
            logging.error(f"Failed to apply conflict resolutions: {e}")
    
    def _get_last_sync_timestamp(self) -> Optional[datetime]:
        """Get last sync timestamp."""
        try:
            sync_state_path = self.xdg_paths.get('state', Path('.')) / 'sync' / 'last_sync.json'
            if sync_state_path.exists():
                with open(sync_state_path, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_sync', ''))
            return None
        except Exception as e:
            logging.error(f"Failed to get last sync timestamp: {e}")
            return None
    
    def _update_last_sync_timestamp(self, timestamp: datetime):
        """Update last sync timestamp."""
        try:
            sync_state_path = self.xdg_paths.get('state', Path('.')) / 'sync' / 'last_sync.json'
            sync_state_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sync_state_path, 'w') as f:
                json.dump({'last_sync': timestamp.isoformat()}, f)
                
        except Exception as e:
            logging.error(f"Failed to update last sync timestamp: {e}")
    
    def _find_local_changes_since(self, since: Optional[datetime]) -> List[Dict[str, Any]]:
        """Find local changes since given timestamp."""
        try:
            changes = []
            
            # For this implementation, we'll look at file modification times
            # In a real implementation, you'd track changes more systematically
            
            memory_dirs = [
                self.xdg_paths.get('data', Path('.')),
                self.xdg_paths.get('cache', Path('.')),
                self.xdg_paths.get('state', Path('.'))
            ]
            
            cutoff_time = since.timestamp() if since else 0
            
            for memory_dir in memory_dirs:
                if memory_dir.exists():
                    for file_path in memory_dir.rglob('*'):
                        if file_path.is_file():
                            if file_path.stat().st_mtime > cutoff_time:
                                changes.append({
                                    'type': 'update',
                                    'path': str(file_path.relative_to(memory_dir)),
                                    'content': self._read_file_safely(file_path),
                                    'metadata': {
                                        'size': file_path.stat().st_size,
                                        'modified': file_path.stat().st_mtime
                                    }
                                })
            
            return changes
            
        except Exception as e:
            logging.error(f"Failed to find local changes: {e}")
            return []
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def _get_machine_id(self) -> str:
        """Get unique machine identifier."""
        try:
            import platform
            machine_info = f"{platform.node()}_{platform.system()}_{platform.machine()}"
            return hashlib.md5(machine_info.encode()).hexdigest()[:12]
        except Exception:
            return str(uuid.uuid4())[:12]
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate content hash."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _get_package_size(self, package: Dict[str, Any]) -> int:
        """Get sync package size."""
        try:
            return len(json.dumps(package).encode('utf-8'))
        except Exception:
            return 0
    
    def _compress_sync_package(self, package: Dict[str, Any]) -> Dict[str, Any]:
        """Compress sync package."""
        try:
            # Simple compression: gzip the JSON
            json_str = json.dumps(package)
            compressed = gzip.compress(json_str.encode('utf-8'))
            
            return {
                'compressed': True,
                'original_size': len(json_str),
                'compressed_size': len(compressed),
                'data': compressed.hex()  # Store as hex string
            }
            
        except Exception as e:
            logging.error(f"Failed to compress sync package: {e}")
            return package
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        try:
            last_sync = self._get_last_sync_timestamp()
            
            return {
                'enabled': self.config.get('sync', {}).get('enabled', False),
                'strategy': self.config.get('sync', {}).get('strategy', 'git'),
                'last_sync': last_sync.isoformat() if last_sync else None,
                'machine_id': self._get_machine_id(),
                'xdg_paths': {k: str(v) for k, v in self.xdg_paths.items()},
                'backend_status': getattr(self.sync_backend, 'get_status', lambda: {})()
            }
            
        except Exception as e:
            logging.error(f"Failed to get sync status: {e}")
            return {'error': str(e)}