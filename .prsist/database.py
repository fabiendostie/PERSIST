#!/usr/bin/env python3
"""
Database module for Prsist Memory System.
Handles SQLite operations for session tracking and context storage.
"""

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

class MemoryDatabase:
    """SQLite database manager for memory system."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = Path(__file__).parent / "storage" / "sessions.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    project_path TEXT NOT NULL,
                    context_data TEXT,
                    status TEXT DEFAULT 'active',
                    git_hash TEXT,
                    git_branch TEXT
                );
                
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data TEXT,
                    output_data TEXT,
                    execution_time_ms INTEGER,
                    success BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                CREATE TABLE IF NOT EXISTS file_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_hash TEXT,
                    line_changes TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                CREATE TABLE IF NOT EXISTS context_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    context_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Git commits tracking
                CREATE TABLE IF NOT EXISTS git_commits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    commit_sha TEXT UNIQUE NOT NULL,
                    session_id TEXT,
                    branch_name TEXT,
                    commit_message TEXT,
                    author_email TEXT,
                    commit_timestamp DATETIME,
                    changed_files_count INTEGER,
                    lines_added INTEGER,
                    lines_deleted INTEGER,
                    memory_impact_score REAL,
                    commit_metadata JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- File changes correlation
                CREATE TABLE IF NOT EXISTS git_file_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    commit_sha TEXT,
                    session_id TEXT,
                    file_path TEXT,
                    change_type TEXT, -- 'added', 'modified', 'deleted', 'renamed'
                    lines_added INTEGER,
                    lines_deleted INTEGER,
                    significance_score REAL,
                    context_summary TEXT,
                    diff_content TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (commit_sha) REFERENCES git_commits (commit_sha),
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Branch memory context
                CREATE TABLE IF NOT EXISTS git_branch_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    branch_name TEXT UNIQUE,
                    base_branch TEXT,
                    context_data JSON,
                    last_updated DATETIME,
                    active_sessions JSON,
                    memory_snapshot JSON,
                    branch_metadata JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Memory-commit relationships
                CREATE TABLE IF NOT EXISTS memory_commit_correlation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    commit_sha TEXT,
                    correlation_type TEXT, -- 'direct', 'related', 'background'
                    correlation_strength REAL,
                    context_overlap JSON,
                    analysis_metadata JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id),
                    FOREIGN KEY (commit_sha) REFERENCES git_commits (commit_sha)
                );
                
                -- Git synchronization tracking
                CREATE TABLE IF NOT EXISTS git_sync_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT,
                    branch_name TEXT,
                    last_sync_commit TEXT,
                    sync_timestamp DATETIME,
                    sync_status TEXT, -- 'pending', 'completed', 'failed', 'conflict'
                    conflict_data JSON,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Documentation generation tracking
                CREATE TABLE IF NOT EXISTS documentation_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    commit_sha TEXT,
                    session_id TEXT,
                    doc_type TEXT, -- 'changelog', 'commit_summary', 'release_note', 'adr'
                    content TEXT,
                    metadata JSON,
                    auto_generated BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (commit_sha) REFERENCES git_commits (commit_sha),
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
                CREATE INDEX IF NOT EXISTS idx_tool_usage_session_id ON tool_usage(session_id);
                CREATE INDEX IF NOT EXISTS idx_file_interactions_session_id ON file_interactions(session_id);
                CREATE INDEX IF NOT EXISTS idx_context_entries_session_id ON context_entries(session_id);
                CREATE INDEX IF NOT EXISTS idx_git_commits_sha ON git_commits(commit_sha);
                CREATE INDEX IF NOT EXISTS idx_git_commits_branch ON git_commits(branch_name);
                CREATE INDEX IF NOT EXISTS idx_git_commits_session ON git_commits(session_id);
                CREATE INDEX IF NOT EXISTS idx_git_file_changes_commit ON git_file_changes(commit_sha);
                
                -- Phase 3 Extensions: Advanced Context Management
                CREATE TABLE IF NOT EXISTS context_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL, -- 'auto', 'manual', 'compression'
                    context_data TEXT NOT NULL,  -- JSON compressed context
                    compression_level INTEGER DEFAULT 0,
                    relevance_scores TEXT,  -- JSON array of relevance scores
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Phase 3: File relevance tracking
                CREATE TABLE IF NOT EXISTS file_relevance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    session_id TEXT,
                    relevance_score REAL NOT NULL,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    relevance_factors TEXT,  -- JSON array of factors
                    expires_at DATETIME,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Phase 3: Dynamic context injection tracking
                CREATE TABLE IF NOT EXISTS context_injections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    injection_type TEXT NOT NULL, -- 'task_relevant', 'pattern_based', 'similarity'
                    injected_content TEXT NOT NULL,  -- JSON content injected
                    relevance_score REAL,
                    injection_reason TEXT,
                    token_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Phase 3: Cross-session memory relationships
                CREATE TABLE IF NOT EXISTS cross_session_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_session_id TEXT NOT NULL,
                    target_session_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL, -- 'similar_task', 'continuation', 'related_problem'
                    relationship_strength REAL DEFAULT 0.5,
                    shared_elements TEXT,  -- JSON array of shared elements
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_session_id) REFERENCES sessions (id),
                    FOREIGN KEY (target_session_id) REFERENCES sessions (id)
                );
                
                -- Phase 3: Memory optimization tracking
                CREATE TABLE IF NOT EXISTS memory_optimizations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    optimization_type TEXT NOT NULL, -- 'compression', 'cleanup', 'relevance_update'
                    before_size INTEGER,
                    after_size INTEGER,
                    optimization_details TEXT,  -- JSON details
                    performance_impact REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Phase 3: Enhanced file change impact tracking
                CREATE TABLE IF NOT EXISTS change_impact_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    impact_score REAL NOT NULL,
                    affected_sessions TEXT,  -- JSON array of session IDs
                    memory_invalidation BOOLEAN DEFAULT FALSE,
                    context_refresh_required BOOLEAN DEFAULT FALSE,
                    analysis_metadata TEXT,  -- JSON analysis details
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Phase 3: Performance metrics
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL, -- 'context_size', 'injection_time', 'relevance_calc_time'
                    metric_value REAL NOT NULL,
                    session_id TEXT,
                    measurement_context TEXT,  -- JSON context
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                );
                
                -- Create Phase 3 indexes
                CREATE INDEX IF NOT EXISTS idx_context_snapshots_session ON context_snapshots(session_id);
                CREATE INDEX IF NOT EXISTS idx_file_relevance_path ON file_relevance(file_path);
                CREATE INDEX IF NOT EXISTS idx_file_relevance_score ON file_relevance(relevance_score);
                CREATE INDEX IF NOT EXISTS idx_context_injections_session ON context_injections(session_id);
                CREATE INDEX IF NOT EXISTS idx_cross_session_source ON cross_session_memory(source_session_id);
                CREATE INDEX IF NOT EXISTS idx_cross_session_target ON cross_session_memory(target_session_id);
                CREATE INDEX IF NOT EXISTS idx_change_impact_file ON change_impact_analysis(file_path);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
                CREATE INDEX IF NOT EXISTS idx_git_file_changes_file ON git_file_changes(file_path);
                CREATE INDEX IF NOT EXISTS idx_git_branch_context_branch ON git_branch_context(branch_name);
                CREATE INDEX IF NOT EXISTS idx_memory_commit_correlation_session ON memory_commit_correlation(session_id);
                CREATE INDEX IF NOT EXISTS idx_memory_commit_correlation_commit ON memory_commit_correlation(commit_sha);
            """)
    
    def create_session(self, session_id: str, project_path: str, 
                      context_data: Dict = None, git_info: Dict = None) -> bool:
        """Create new session record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sessions (id, project_path, context_data, git_hash, git_branch)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    str(project_path),
                    json.dumps(context_data) if context_data else None,
                    git_info.get('hash') if git_info else None,
                    git_info.get('branch') if git_info else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to create session {session_id}: {e}")
            return False
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session with new data."""
        try:
            updates = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['context_data', 'status', 'git_hash', 'git_branch']:
                    updates.append(f"{key} = ?")
                    if key == 'context_data' and isinstance(value, dict):
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
            
            if not updates:
                return True
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            values.append(session_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"""
                    UPDATE sessions SET {', '.join(updates)}
                    WHERE id = ?
                """, values)
                return True
        except Exception as e:
            logging.error(f"Failed to update session {session_id}: {e}")
            return False
    
    def log_tool_usage(self, session_id: str, tool_name: str, 
                      input_data: Any = None, output_data: Any = None,
                      execution_time_ms: int = None, success: bool = True) -> bool:
        """Log tool usage for session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO tool_usage 
                    (session_id, tool_name, input_data, output_data, execution_time_ms, success)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    tool_name,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    execution_time_ms,
                    success
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to log tool usage for {session_id}: {e}")
            return False
    
    def log_file_interaction(self, session_id: str, file_path: str, 
                           action_type: str, content_hash: str = None,
                           line_changes: Dict = None) -> bool:
        """Log file interaction for session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO file_interactions 
                    (session_id, file_path, action_type, content_hash, line_changes)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    str(file_path),
                    action_type,
                    content_hash,
                    json.dumps(line_changes) if line_changes else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to log file interaction for {session_id}: {e}")
            return False
    
    def add_context_entry(self, session_id: str, context_type: str, 
                         content: str, relevance_score: float = 1.0) -> bool:
        """Add context entry for session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_entries 
                    (session_id, context_type, content, relevance_score)
                    VALUES (?, ?, ?, ?)
                """, (session_id, context_type, content, relevance_score))
                return True
        except Exception as e:
            logging.error(f"Failed to add context entry for {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM sessions WHERE id = ?
                """, (session_id,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    if result['context_data']:
                        result['context_data'] = json.loads(result['context_data'])
                    return result
                return None
        except Exception as e:
            logging.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                sessions = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['context_data']:
                        result['context_data'] = json.loads(result['context_data'])
                    sessions.append(result)
                return sessions
        except Exception as e:
            logging.error(f"Failed to get recent sessions: {e}")
            return []
    
    def get_session_tool_usage(self, session_id: str) -> List[Dict]:
        """Get tool usage for session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM tool_usage 
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logging.error(f"Failed to get tool usage for {session_id}: {e}")
            return []
    
    def cleanup_old_sessions(self, retention_days: int = 30) -> int:
        """Clean up old sessions and related data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM sessions 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(retention_days))
                return cursor.rowcount
        except Exception as e:
            logging.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    # Git-specific methods
    
    def record_commit(self, commit_sha: str, session_id: str = None, 
                     branch_name: str = None, commit_message: str = None,
                     author_email: str = None, commit_timestamp: str = None,
                     changed_files_count: int = None, lines_added: int = None,
                     lines_deleted: int = None, memory_impact_score: float = None,
                     commit_metadata: Dict = None) -> bool:
        """Record a git commit in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO git_commits 
                    (commit_sha, session_id, branch_name, commit_message, author_email, 
                     commit_timestamp, changed_files_count, lines_added, lines_deleted, 
                     memory_impact_score, commit_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    commit_sha, session_id, branch_name, commit_message, author_email,
                    commit_timestamp, changed_files_count, lines_added, lines_deleted,
                    memory_impact_score, json.dumps(commit_metadata) if commit_metadata else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record commit {commit_sha}: {e}")
            return False
    
    def record_file_change(self, commit_sha: str, session_id: str = None,
                          file_path: str = None, change_type: str = None,
                          lines_added: int = None, lines_deleted: int = None,
                          significance_score: float = None, context_summary: str = None,
                          diff_content: str = None) -> bool:
        """Record a file change for a commit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO git_file_changes 
                    (commit_sha, session_id, file_path, change_type, lines_added, 
                     lines_deleted, significance_score, context_summary, diff_content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    commit_sha, session_id, file_path, change_type, lines_added,
                    lines_deleted, significance_score, context_summary, diff_content
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record file change for {commit_sha}: {e}")
            return False
    
    def update_branch_context(self, branch_name: str, base_branch: str = None,
                             context_data: Dict = None, active_sessions: List = None,
                             memory_snapshot: Dict = None, branch_metadata: Dict = None) -> bool:
        """Update or create branch context."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO git_branch_context 
                    (branch_name, base_branch, context_data, last_updated, 
                     active_sessions, memory_snapshot, branch_metadata)
                    VALUES (?, ?, ?, datetime('now'), ?, ?, ?)
                """, (
                    branch_name, base_branch,
                    json.dumps(context_data) if context_data else None,
                    json.dumps(active_sessions) if active_sessions else None,
                    json.dumps(memory_snapshot) if memory_snapshot else None,
                    json.dumps(branch_metadata) if branch_metadata else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to update branch context for {branch_name}: {e}")
            return False
    
    def create_commit_correlation(self, session_id: str, commit_sha: str,
                                 correlation_type: str, correlation_strength: float,
                                 context_overlap: Dict = None, analysis_metadata: Dict = None) -> bool:
        """Create a correlation between a session and commit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memory_commit_correlation 
                    (session_id, commit_sha, correlation_type, correlation_strength, 
                     context_overlap, analysis_metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id, commit_sha, correlation_type, correlation_strength,
                    json.dumps(context_overlap) if context_overlap else None,
                    json.dumps(analysis_metadata) if analysis_metadata else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to create correlation for {session_id}-{commit_sha}: {e}")
            return False
    
    def get_commit_by_sha(self, commit_sha: str) -> Optional[Dict]:
        """Get commit information by SHA."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM git_commits WHERE commit_sha = ?
                """, (commit_sha,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    if result['commit_metadata']:
                        result['commit_metadata'] = json.loads(result['commit_metadata'])
                    return result
                return None
        except Exception as e:
            logging.error(f"Failed to get commit {commit_sha}: {e}")
            return None
    
    def get_branch_context(self, branch_name: str) -> Optional[Dict]:
        """Get branch context information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM git_branch_context WHERE branch_name = ?
                """, (branch_name,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    for json_field in ['context_data', 'active_sessions', 'memory_snapshot', 'branch_metadata']:
                        if result[json_field]:
                            result[json_field] = json.loads(result[json_field])
                    return result
                return None
        except Exception as e:
            logging.error(f"Failed to get branch context for {branch_name}: {e}")
            return None
    
    def get_recent_commits(self, branch_name: str = None, limit: int = 10) -> List[Dict]:
        """Get recent commits, optionally filtered by branch."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if branch_name:
                    cursor = conn.execute("""
                        SELECT * FROM git_commits 
                        WHERE branch_name = ?
                        ORDER BY commit_timestamp DESC 
                        LIMIT ?
                    """, (branch_name, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM git_commits 
                        ORDER BY commit_timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                commits = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['commit_metadata']:
                        result['commit_metadata'] = json.loads(result['commit_metadata'])
                    commits.append(result)
                return commits
        except Exception as e:
            logging.error(f"Failed to get recent commits: {e}")
            return []
    
    def get_session_commits(self, session_id: str) -> List[Dict]:
        """Get all commits associated with a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM git_commits 
                    WHERE session_id = ?
                    ORDER BY commit_timestamp ASC
                """, (session_id,))
                
                commits = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['commit_metadata']:
                        result['commit_metadata'] = json.loads(result['commit_metadata'])
                    commits.append(result)
                return commits
        except Exception as e:
            logging.error(f"Failed to get session commits for {session_id}: {e}")
            return []
    
    def record_documentation_entry(self, commit_sha: str = None, session_id: str = None,
                                  doc_type: str = None, content: str = None,
                                  metadata: Dict = None, auto_generated: bool = True) -> bool:
        """Record a documentation entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO documentation_entries 
                    (commit_sha, session_id, doc_type, content, metadata, auto_generated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    commit_sha, session_id, doc_type, content,
                    json.dumps(metadata) if metadata else None, auto_generated
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record documentation entry: {e}")
            return False
    
    # Phase 3 Methods: Advanced Context Management
    
    def create_context_snapshot(self, session_id: str, snapshot_type: str,
                               context_data: Dict, compression_level: int = 0,
                               relevance_scores: List = None) -> bool:
        """Create a context snapshot."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_snapshots 
                    (session_id, snapshot_type, context_data, compression_level, relevance_scores)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id, snapshot_type, json.dumps(context_data),
                    compression_level, json.dumps(relevance_scores) if relevance_scores else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to create context snapshot: {e}")
            return False
    
    def get_context_snapshots(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get context snapshots for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM context_snapshots 
                    WHERE session_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (session_id, limit))
                
                snapshots = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['context_data'] = json.loads(result['context_data'])
                    if result['relevance_scores']:
                        result['relevance_scores'] = json.loads(result['relevance_scores'])
                    snapshots.append(result)
                return snapshots
        except Exception as e:
            logging.error(f"Failed to get context snapshots: {e}")
            return []
    
    def update_file_relevance(self, file_path: str, session_id: str = None,
                             relevance_score: float = 0.0, relevance_factors: List = None,
                             expires_at: datetime = None) -> bool:
        """Update file relevance score."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if entry exists
                existing = conn.execute("""
                    SELECT * FROM file_relevance 
                    WHERE file_path = ? AND (session_id = ? OR session_id IS NULL)
                """, (file_path, session_id)).fetchone()
                
                if existing:
                    # Update existing entry
                    conn.execute("""
                        UPDATE file_relevance 
                        SET relevance_score = ?, access_count = access_count + 1,
                            last_accessed = datetime('now'), relevance_factors = ?,
                            expires_at = ?
                        WHERE file_path = ? AND (session_id = ? OR session_id IS NULL)
                    """, (
                        relevance_score, json.dumps(relevance_factors) if relevance_factors else None,
                        expires_at.isoformat() if expires_at else None, file_path, session_id
                    ))
                else:
                    # Create new entry
                    conn.execute("""
                        INSERT INTO file_relevance 
                        (file_path, session_id, relevance_score, relevance_factors, expires_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        file_path, session_id, relevance_score,
                        json.dumps(relevance_factors) if relevance_factors else None,
                        expires_at.isoformat() if expires_at else None
                    ))
                return True
        except Exception as e:
            logging.error(f"Failed to update file relevance: {e}")
            return False
    
    def get_file_relevance(self, file_path: str = None, session_id: str = None,
                          min_score: float = 0.0, limit: int = 50) -> List[Dict]:
        """Get file relevance scores."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if file_path:
                    cursor = conn.execute("""
                        SELECT * FROM file_relevance 
                        WHERE file_path = ? AND relevance_score >= ?
                        ORDER BY relevance_score DESC
                    """, (file_path, min_score))
                elif session_id:
                    cursor = conn.execute("""
                        SELECT * FROM file_relevance 
                        WHERE session_id = ? AND relevance_score >= ?
                        ORDER BY relevance_score DESC 
                        LIMIT ?
                    """, (session_id, min_score, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM file_relevance 
                        WHERE relevance_score >= ? AND (expires_at IS NULL OR expires_at > datetime('now'))
                        ORDER BY relevance_score DESC 
                        LIMIT ?
                    """, (min_score, limit))
                
                relevances = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['relevance_factors']:
                        result['relevance_factors'] = json.loads(result['relevance_factors'])
                    relevances.append(result)
                return relevances
        except Exception as e:
            logging.error(f"Failed to get file relevance: {e}")
            return []
    
    def record_context_injection(self, session_id: str, injection_type: str,
                                injected_content: Dict, relevance_score: float = None,
                                injection_reason: str = None, token_count: int = None) -> bool:
        """Record a context injection event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO context_injections 
                    (session_id, injection_type, injected_content, relevance_score, 
                     injection_reason, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id, injection_type, json.dumps(injected_content),
                    relevance_score, injection_reason, token_count
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record context injection: {e}")
            return False
    
    def get_context_injections(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get context injections for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM context_injections 
                    WHERE session_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (session_id, limit))
                
                injections = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['injected_content'] = json.loads(result['injected_content'])
                    injections.append(result)
                return injections
        except Exception as e:
            logging.error(f"Failed to get context injections: {e}")
            return []
    
    def create_cross_session_relationship(self, source_session_id: str, target_session_id: str,
                                        relationship_type: str, relationship_strength: float,
                                        shared_elements: List = None) -> bool:
        """Create a cross-session memory relationship."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cross_session_memory 
                    (source_session_id, target_session_id, relationship_type, 
                     relationship_strength, shared_elements)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source_session_id, target_session_id, relationship_type,
                    relationship_strength, json.dumps(shared_elements) if shared_elements else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to create cross-session relationship: {e}")
            return False
    
    def get_related_sessions(self, session_id: str, min_strength: float = 0.5,
                           limit: int = 10) -> List[Dict]:
        """Get sessions related to the given session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM cross_session_memory 
                    WHERE (source_session_id = ? OR target_session_id = ?)
                        AND relationship_strength >= ?
                    ORDER BY relationship_strength DESC 
                    LIMIT ?
                """, (session_id, session_id, min_strength, limit))
                
                relationships = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['shared_elements']:
                        result['shared_elements'] = json.loads(result['shared_elements'])
                    relationships.append(result)
                return relationships
        except Exception as e:
            logging.error(f"Failed to get related sessions: {e}")
            return []
    
    def record_memory_optimization(self, session_id: str, optimization_type: str,
                                  before_size: int = None, after_size: int = None,
                                  optimization_details: Dict = None, performance_impact: float = None) -> bool:
        """Record a memory optimization event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memory_optimizations 
                    (session_id, optimization_type, before_size, after_size, 
                     optimization_details, performance_impact)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id, optimization_type, before_size, after_size,
                    json.dumps(optimization_details) if optimization_details else None,
                    performance_impact
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record memory optimization: {e}")
            return False
    
    def record_change_impact(self, file_path: str, change_type: str, impact_score: float,
                           affected_sessions: List = None, memory_invalidation: bool = False,
                           context_refresh_required: bool = False, analysis_metadata: Dict = None) -> bool:
        """Record file change impact analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO change_impact_analysis 
                    (file_path, change_type, impact_score, affected_sessions, 
                     memory_invalidation, context_refresh_required, analysis_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path, change_type, impact_score,
                    json.dumps(affected_sessions) if affected_sessions else None,
                    memory_invalidation, context_refresh_required,
                    json.dumps(analysis_metadata) if analysis_metadata else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record change impact: {e}")
            return False
    
    def get_change_impacts(self, file_path: str = None, min_impact: float = 0.5,
                          limit: int = 50) -> List[Dict]:
        """Get file change impact analyses."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if file_path:
                    cursor = conn.execute("""
                        SELECT * FROM change_impact_analysis 
                        WHERE file_path = ? AND impact_score >= ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (file_path, min_impact, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM change_impact_analysis 
                        WHERE impact_score >= ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (min_impact, limit))
                
                impacts = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['affected_sessions']:
                        result['affected_sessions'] = json.loads(result['affected_sessions'])
                    if result['analysis_metadata']:
                        result['analysis_metadata'] = json.loads(result['analysis_metadata'])
                    impacts.append(result)
                return impacts
        except Exception as e:
            logging.error(f"Failed to get change impacts: {e}")
            return []
    
    def record_performance_metric(self, metric_type: str, metric_value: float,
                                 session_id: str = None, measurement_context: Dict = None) -> bool:
        """Record a performance metric."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (metric_type, metric_value, session_id, measurement_context)
                    VALUES (?, ?, ?, ?)
                """, (
                    metric_type, metric_value, session_id,
                    json.dumps(measurement_context) if measurement_context else None
                ))
                return True
        except Exception as e:
            logging.error(f"Failed to record performance metric: {e}")
            return False
    
    def get_performance_metrics(self, metric_type: str = None, session_id: str = None,
                               limit: int = 100) -> List[Dict]:
        """Get performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if metric_type and session_id:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        WHERE metric_type = ? AND session_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (metric_type, session_id, limit))
                elif metric_type:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        WHERE metric_type = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (metric_type, limit))
                elif session_id:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        WHERE session_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (session_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                metrics = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['measurement_context']:
                        result['measurement_context'] = json.loads(result['measurement_context'])
                    metrics.append(result)
                return metrics
        except Exception as e:
            logging.error(f"Failed to get performance metrics: {e}")
            return []
    
    def get_active_sessions(self, limit: int = 10) -> List[Dict]:
        """Get active sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM sessions 
                    WHERE status = 'active'
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    result = dict(row)
                    if result['context_data']:
                        result['context_data'] = json.loads(result['context_data'])
                    sessions.append(result)
                return sessions
        except Exception as e:
            logging.error(f"Failed to get active sessions: {e}")
            return []
    
    def cleanup_expired_relevance(self) -> int:
        """Clean up expired file relevance entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM file_relevance 
                    WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')
                """)
                return cursor.rowcount
        except Exception as e:
            logging.error(f"Failed to cleanup expired relevance: {e}")
            return 0