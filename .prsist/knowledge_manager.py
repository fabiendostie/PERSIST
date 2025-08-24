#!/usr/bin/env python3
"""
Cross-session knowledge persistence for Prsist Memory System Phase 3.
Manages long-term learning and knowledge accumulation across sessions.
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
import hashlib

from database import MemoryDatabase
from utils import setup_logging

class PatternDetector:
    """Detects patterns in session data."""
    
    def __init__(self):
        """Initialize pattern detector."""
        self.pattern_types = {
            'code': self._detect_code_patterns,
            'workflow': self._detect_workflow_patterns,
            'decision': self._detect_decision_patterns,
            'problem': self._detect_problem_patterns
        }
    
    def extract_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from session data."""
        try:
            patterns = []
            
            for pattern_type, detector in self.pattern_types.items():
                detected = detector(session_data)
                for pattern in detected:
                    pattern['type'] = pattern_type
                    pattern['detected_at'] = datetime.now().isoformat()
                    pattern['session_id'] = session_data.get('session_id', 'unknown')
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Failed to extract patterns: {e}")
            return []
    
    def _detect_code_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code-related patterns."""
        patterns = []
        
        try:
            # Analyze tool usage for code patterns
            tool_usage = session_data.get('tool_usage', [])
            
            # Pattern: Frequent file editing
            edit_tools = [t for t in tool_usage if t.get('tool_name') in ['Edit', 'MultiEdit', 'Write']]
            if len(edit_tools) > 10:
                patterns.append({
                    'pattern_id': 'heavy_editing',
                    'description': 'Heavy file editing session',
                    'frequency': len(edit_tools),
                    'files_involved': list(set(t.get('file_path', '') for t in edit_tools if t.get('file_path'))),
                    'confidence': min(1.0, len(edit_tools) / 20.0)
                })
            
            # Pattern: Test-driven development
            test_reads = [t for t in tool_usage if 'test' in t.get('file_path', '').lower()]
            code_edits = [t for t in edit_tools if 'test' not in t.get('file_path', '').lower()]
            
            if len(test_reads) > 3 and len(code_edits) > 3:
                patterns.append({
                    'pattern_id': 'tdd_workflow',
                    'description': 'Test-driven development workflow',
                    'test_interactions': len(test_reads),
                    'code_edits': len(code_edits),
                    'confidence': min(1.0, (len(test_reads) + len(code_edits)) / 20.0)
                })
            
            # Pattern: Configuration changes
            config_files = [t for t in tool_usage if any(ext in t.get('file_path', '') for ext in ['.yaml', '.json', '.toml', '.ini'])]
            if len(config_files) > 5:
                patterns.append({
                    'pattern_id': 'configuration_focus',
                    'description': 'Configuration-heavy session',
                    'config_interactions': len(config_files),
                    'config_files': list(set(t.get('file_path', '') for t in config_files if t.get('file_path'))),
                    'confidence': min(1.0, len(config_files) / 10.0)
                })
            
        except Exception as e:
            logging.error(f"Failed to detect code patterns: {e}")
        
        return patterns
    
    def _detect_workflow_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect workflow patterns."""
        patterns = []
        
        try:
            tool_usage = session_data.get('tool_usage', [])
            
            # Pattern: Research workflow
            read_tools = [t for t in tool_usage if t.get('tool_name') == 'Read']
            search_tools = [t for t in tool_usage if t.get('tool_name') in ['Grep', 'Glob']]
            
            if len(read_tools) > 10 or len(search_tools) > 5:
                patterns.append({
                    'pattern_id': 'research_workflow',
                    'description': 'Research and exploration workflow',
                    'read_operations': len(read_tools),
                    'search_operations': len(search_tools),
                    'files_explored': len(set(t.get('file_path', '') for t in read_tools if t.get('file_path'))),
                    'confidence': min(1.0, (len(read_tools) + len(search_tools)) / 15.0)
                })
            
            # Pattern: Implementation workflow
            edit_tools = [t for t in tool_usage if t.get('tool_name') in ['Edit', 'MultiEdit', 'Write']]
            bash_tools = [t for t in tool_usage if t.get('tool_name') == 'Bash']
            
            if len(edit_tools) > 5 and len(bash_tools) > 3:
                patterns.append({
                    'pattern_id': 'implementation_workflow',
                    'description': 'Active implementation workflow',
                    'edit_operations': len(edit_tools),
                    'bash_operations': len(bash_tools),
                    'confidence': min(1.0, (len(edit_tools) + len(bash_tools)) / 15.0)
                })
            
        except Exception as e:
            logging.error(f"Failed to detect workflow patterns: {e}")
        
        return patterns
    
    def _detect_decision_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect decision-making patterns."""
        patterns = []
        
        try:
            # Analyze context data for decision indicators
            context_data = session_data.get('context_data', {})
            
            # Look for decision keywords in context
            decision_keywords = ['decide', 'choose', 'option', 'alternative', 'approach', 'strategy']
            context_text = json.dumps(context_data, default=str).lower()
            
            decision_indicators = sum(1 for keyword in decision_keywords if keyword in context_text)
            
            if decision_indicators > 2:
                patterns.append({
                    'pattern_id': 'decision_making',
                    'description': 'Decision-making session',
                    'decision_indicators': decision_indicators,
                    'confidence': min(1.0, decision_indicators / 5.0)
                })
            
        except Exception as e:
            logging.error(f"Failed to detect decision patterns: {e}")
        
        return patterns
    
    def _detect_problem_patterns(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect problem-solving patterns."""
        patterns = []
        
        try:
            tool_usage = session_data.get('tool_usage', [])
            
            # Pattern: Debugging session
            error_keywords = ['error', 'exception', 'fail', 'bug', 'issue', 'problem']
            bash_tools = [t for t in tool_usage if t.get('tool_name') == 'Bash']
            
            error_related_tools = []
            for tool in tool_usage:
                tool_data = json.dumps(tool, default=str).lower()
                if any(keyword in tool_data for keyword in error_keywords):
                    error_related_tools.append(tool)
            
            if len(error_related_tools) > 3 or len(bash_tools) > 5:
                patterns.append({
                    'pattern_id': 'debugging_session',
                    'description': 'Problem-solving/debugging session',
                    'error_related_operations': len(error_related_tools),
                    'bash_operations': len(bash_tools),
                    'confidence': min(1.0, (len(error_related_tools) + len(bash_tools)) / 10.0)
                })
            
        except Exception as e:
            logging.error(f"Failed to detect problem patterns: {e}")
        
        return patterns


class DecisionTracker:
    """Tracks important decisions made during sessions."""
    
    def __init__(self):
        """Initialize decision tracker."""
        self.decision_indicators = [
            'decided to', 'chose to', 'selected', 'opted for', 'went with',
            'architecture', 'approach', 'strategy', 'solution', 'implementation'
        ]
    
    def extract_decisions(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract important decisions from session data."""
        try:
            decisions = []
            
            # Analyze context data for decisions
            context_decisions = self._extract_from_context(session_data.get('context_data', {}))
            decisions.extend(context_decisions)
            
            # Analyze tool usage for implementation decisions
            tool_decisions = self._extract_from_tool_usage(session_data.get('tool_usage', []))
            decisions.extend(tool_decisions)
            
            return decisions
            
        except Exception as e:
            logging.error(f"Failed to extract decisions: {e}")
            return []
    
    def _extract_from_context(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract decisions from context data."""
        decisions = []
        
        try:
            # Look for explicit decision records
            if 'decisions' in context_data:
                for decision in context_data['decisions']:
                    decisions.append({
                        'decision_id': self._generate_decision_id(decision),
                        'type': 'explicit',
                        'summary': decision.get('summary', ''),
                        'rationale': decision.get('rationale', ''),
                        'alternatives': decision.get('alternatives', []),
                        'impact': decision.get('impact', 'medium'),
                        'timestamp': decision.get('timestamp', datetime.now().isoformat())
                    })
            
            # Look for implicit decisions in notes or descriptions
            text_fields = ['notes', 'description', 'summary', 'goals']
            for field in text_fields:
                if field in context_data:
                    implicit_decisions = self._extract_implicit_decisions(context_data[field])
                    decisions.extend(implicit_decisions)
            
        except Exception as e:
            logging.error(f"Failed to extract context decisions: {e}")
        
        return decisions
    
    def _extract_from_tool_usage(self, tool_usage: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract decisions from tool usage patterns."""
        decisions = []
        
        try:
            # Look for architectural decisions in file patterns
            config_changes = [t for t in tool_usage if any(ext in t.get('file_path', '') for ext in ['.yaml', '.json', '.toml'])]
            
            if len(config_changes) > 3:
                decisions.append({
                    'decision_id': f"config_change_{datetime.now().timestamp()}",
                    'type': 'configuration',
                    'summary': f'Configuration changes across {len(config_changes)} files',
                    'rationale': 'System configuration adjustments',
                    'files_affected': list(set(t.get('file_path', '') for t in config_changes if t.get('file_path'))),
                    'impact': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Look for structural decisions in code changes
            code_edits = [t for t in tool_usage if t.get('tool_name') in ['Edit', 'MultiEdit', 'Write']]
            if len(code_edits) > 10:
                unique_files = set(t.get('file_path', '') for t in code_edits if t.get('file_path'))
                if len(unique_files) > 5:
                    decisions.append({
                        'decision_id': f"structural_change_{datetime.now().timestamp()}",
                        'type': 'structural',
                        'summary': f'Structural changes across {len(unique_files)} files',
                        'rationale': 'Code structure modifications',
                        'files_affected': list(unique_files),
                        'impact': 'high',
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logging.error(f"Failed to extract tool usage decisions: {e}")
        
        return decisions
    
    def _extract_implicit_decisions(self, text: str) -> List[Dict[str, Any]]:
        """Extract implicit decisions from text."""
        decisions = []
        
        try:
            if not isinstance(text, str):
                text = str(text)
            
            # Look for decision indicators
            for indicator in self.decision_indicators:
                pattern = rf'{indicator}\s+([^.!?]+)'
                matches = re.finditer(pattern, text.lower())
                
                for match in matches:
                    decision_text = match.group(1).strip()
                    if len(decision_text) > 10:  # Ensure meaningful content
                        decisions.append({
                            'decision_id': self._generate_decision_id(decision_text),
                            'type': 'implicit',
                            'summary': decision_text,
                            'rationale': f'Extracted from: "{indicator} {decision_text}"',
                            'confidence': 0.7,
                            'timestamp': datetime.now().isoformat()
                        })
            
        except Exception as e:
            logging.error(f"Failed to extract implicit decisions: {e}")
        
        return decisions
    
    def _generate_decision_id(self, decision_content: Any) -> str:
        """Generate unique ID for a decision."""
        content_str = json.dumps(decision_content, default=str, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:12]


class KnowledgeManager:
    """Manages persistent knowledge across sessions."""
    
    def __init__(self, storage_path: str):
        """Initialize knowledge manager."""
        self.storage_path = Path(storage_path)
        self.knowledge_db = self.init_knowledge_database()
        self.pattern_detector = PatternDetector()
        self.decision_tracker = DecisionTracker()
    
    def init_knowledge_database(self) -> sqlite3.Connection:
        """Initialize knowledge database with extended schema."""
        try:
            db_path = self.storage_path / 'knowledge.db'
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            # Create knowledge persistence tables
            conn.executescript("""
                -- Cross-session knowledge patterns
                CREATE TABLE IF NOT EXISTS knowledge_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,  -- JSON data
                    frequency INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 0.5,
                    contexts TEXT,  -- JSON array of contexts
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.5
                );
                
                -- Decision tracking across sessions
                CREATE TABLE IF NOT EXISTS project_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_type TEXT NOT NULL,
                    decision_summary TEXT NOT NULL,
                    rationale TEXT,
                    alternatives_considered TEXT,  -- JSON array
                    outcome TEXT,
                    session_ids TEXT,  -- JSON array
                    impact_score REAL DEFAULT 0.5,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    decision_id TEXT UNIQUE
                );
                
                -- Long-term project memory
                CREATE TABLE IF NOT EXISTS project_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    context TEXT,  -- JSON data
                    importance_score REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    last_accessed DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Cross-session relationships
                CREATE TABLE IF NOT EXISTS session_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_1 TEXT NOT NULL,
                    session_2 TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_strength REAL DEFAULT 0.5,
                    shared_context TEXT,  -- JSON data
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Knowledge evolution tracking
                CREATE TABLE IF NOT EXISTS knowledge_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_item_id INTEGER,
                    change_type TEXT NOT NULL,  -- 'created', 'updated', 'reinforced', 'deprecated'
                    change_description TEXT,
                    session_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON knowledge_patterns(pattern_type);
                CREATE INDEX IF NOT EXISTS idx_patterns_last_seen ON knowledge_patterns(last_seen);
                CREATE INDEX IF NOT EXISTS idx_decisions_type ON project_decisions(decision_type);
                CREATE INDEX IF NOT EXISTS idx_memory_type ON project_memory(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON project_memory(importance_score);
                CREATE INDEX IF NOT EXISTS idx_relationships_sessions ON session_relationships(session_1, session_2);
            """)
            
            conn.commit()
            logging.info("Knowledge database initialized successfully")
            return conn
            
        except Exception as e:
            logging.error(f"Failed to initialize knowledge database: {e}")
            raise
    
    def persist_session_knowledge(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and persist valuable knowledge from session."""
        try:
            session_id = session_data.get('session_id', 'unknown')
            logging.info(f"Persisting knowledge from session {session_id}")
            
            results = {
                'patterns_stored': 0,
                'decisions_stored': 0,
                'relationships_created': 0,
                'memory_entries_created': 0
            }
            
            # Extract and store patterns
            patterns = self.pattern_detector.extract_patterns(session_data)
            for pattern in patterns:
                if self.store_pattern(pattern):
                    results['patterns_stored'] += 1
            
            # Extract and store decisions
            decisions = self.decision_tracker.extract_decisions(session_data)
            for decision in decisions:
                if self.store_decision(decision, session_id):
                    results['decisions_stored'] += 1
            
            # Update project context
            self.update_project_context(session_data)
            
            # Create session relationships
            relationships = self.identify_session_relationships(session_data)
            for relationship in relationships:
                if self.store_session_relationship(relationship):
                    results['relationships_created'] += 1
            
            # Extract general memory entries
            memory_entries = self.extract_memory_entries(session_data)
            for entry in memory_entries:
                if self.store_memory_entry(entry):
                    results['memory_entries_created'] += 1
            
            logging.info(f"Knowledge persistence complete: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Failed to persist session knowledge: {e}")
            return {'error': str(e)}
    
    def store_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Store or update a knowledge pattern."""
        try:
            pattern_id = pattern.get('pattern_id', '')
            pattern_type = pattern.get('type', 'unknown')
            
            # Check if pattern already exists
            existing = self.knowledge_db.execute(
                "SELECT * FROM knowledge_patterns WHERE pattern_data LIKE ? AND pattern_type = ?",
                (f'%{pattern_id}%', pattern_type)
            ).fetchone()
            
            if existing:
                # Update existing pattern
                new_frequency = existing['frequency'] + 1
                self.knowledge_db.execute("""
                    UPDATE knowledge_patterns 
                    SET frequency = ?, last_seen = ?, confidence = ?
                    WHERE id = ?
                """, (new_frequency, datetime.now().isoformat(), 
                         pattern.get('confidence', 0.5), existing['id']))
            else:
                # Create new pattern
                self.knowledge_db.execute("""
                    INSERT INTO knowledge_patterns 
                    (pattern_type, pattern_data, frequency, confidence, contexts)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    pattern_type,
                    json.dumps(pattern),
                    1,
                    pattern.get('confidence', 0.5),
                    json.dumps([pattern.get('session_id', '')])
                ))
            
            self.knowledge_db.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to store pattern: {e}")
            return False
    
    def store_decision(self, decision: Dict[str, Any], session_id: str) -> bool:
        """Store a project decision."""
        try:
            decision_id = decision.get('decision_id', '')
            
            # Check if decision already exists
            existing = self.knowledge_db.execute(
                "SELECT * FROM project_decisions WHERE decision_id = ?",
                (decision_id,)
            ).fetchone()
            
            if existing:
                # Update session list
                session_ids = json.loads(existing['session_ids'] or '[]')
                if session_id not in session_ids:
                    session_ids.append(session_id)
                
                self.knowledge_db.execute("""
                    UPDATE project_decisions 
                    SET session_ids = ?
                    WHERE decision_id = ?
                """, (json.dumps(session_ids), decision_id))
            else:
                # Create new decision
                self.knowledge_db.execute("""
                    INSERT INTO project_decisions 
                    (decision_type, decision_summary, rationale, alternatives_considered,
                     session_ids, impact_score, decision_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.get('type', 'unknown'),
                    decision.get('summary', ''),
                    decision.get('rationale', ''),
                    json.dumps(decision.get('alternatives', [])),
                    json.dumps([session_id]),
                    decision.get('impact_score', 0.5),
                    decision_id
                ))
            
            self.knowledge_db.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to store decision: {e}")
            return False
    
    def update_project_context(self, session_data: Dict[str, Any]):
        """Update project evolution context."""
        try:
            session_id = session_data.get('session_id', 'unknown')
            
            # Store high-level session insights
            insights = self.extract_session_insights(session_data)
            
            for insight in insights:
                self.knowledge_db.execute("""
                    INSERT INTO project_memory 
                    (memory_type, content, context, importance_score)
                    VALUES (?, ?, ?, ?)
                """, (
                    'session_insight',
                    insight['content'],
                    json.dumps({'session_id': session_id, 'context': insight.get('context', {})}),
                    insight.get('importance', 0.5)
                ))
            
            self.knowledge_db.commit()
            
        except Exception as e:
            logging.error(f"Failed to update project context: {e}")
    
    def extract_session_insights(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract high-level insights from session."""
        insights = []
        
        try:
            tool_usage = session_data.get('tool_usage', [])
            
            # Insight: Heavy development activity
            edit_count = len([t for t in tool_usage if t.get('tool_name') in ['Edit', 'MultiEdit', 'Write']])
            if edit_count > 10:
                insights.append({
                    'content': f'Heavy development session with {edit_count} file modifications',
                    'importance': min(1.0, edit_count / 20.0),
                    'context': {'tool_usage_pattern': 'heavy_editing'}
                })
            
            # Insight: Research-heavy session
            read_count = len([t for t in tool_usage if t.get('tool_name') == 'Read'])
            if read_count > 15:
                insights.append({
                    'content': f'Research-intensive session with {read_count} file reads',
                    'importance': min(1.0, read_count / 30.0),
                    'context': {'tool_usage_pattern': 'research_heavy'}
                })
            
        except Exception as e:
            logging.error(f"Failed to extract session insights: {e}")
        
        return insights
    
    def identify_session_relationships(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relationships with other sessions."""
        relationships = []
        
        try:
            current_session_id = session_data.get('session_id', 'unknown')
            
            # Find sessions with similar patterns
            current_patterns = self.pattern_detector.extract_patterns(session_data)
            
            if current_patterns:
                # This would typically query for sessions with similar patterns
                # For now, return empty list
                pass
            
        except Exception as e:
            logging.error(f"Failed to identify session relationships: {e}")
        
        return relationships
    
    def store_session_relationship(self, relationship: Dict[str, Any]) -> bool:
        """Store a session relationship."""
        try:
            self.knowledge_db.execute("""
                INSERT INTO session_relationships 
                (session_1, session_2, relationship_type, relationship_strength, shared_context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                relationship['session_1'],
                relationship['session_2'],
                relationship['type'],
                relationship['strength'],
                json.dumps(relationship.get('context', {}))
            ))
            
            self.knowledge_db.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to store session relationship: {e}")
            return False
    
    def extract_memory_entries(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract general memory entries from session."""
        entries = []
        
        try:
            # Extract lessons learned
            context_data = session_data.get('context_data', {})
            
            if 'lessons_learned' in context_data:
                for lesson in context_data['lessons_learned']:
                    entries.append({
                        'type': 'lesson',
                        'content': lesson,
                        'importance': 0.7
                    })
            
            # Extract preferences
            if 'preferences' in context_data:
                for pref in context_data['preferences']:
                    entries.append({
                        'type': 'preference',
                        'content': pref,
                        'importance': 0.5
                    })
            
        except Exception as e:
            logging.error(f"Failed to extract memory entries: {e}")
        
        return entries
    
    def store_memory_entry(self, entry: Dict[str, Any]) -> bool:
        """Store a general memory entry."""
        try:
            self.knowledge_db.execute("""
                INSERT INTO project_memory 
                (memory_type, content, importance_score)
                VALUES (?, ?, ?)
            """, (
                entry.get('type', 'general'),
                entry.get('content', ''),
                entry.get('importance', 0.5)
            ))
            
            self.knowledge_db.commit()
            return True
            
        except Exception as e:
            logging.error(f"Failed to store memory entry: {e}")
            return False
    
    def get_relevant_knowledge(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant cross-session knowledge."""
        try:
            # Get similar patterns from past sessions
            relevant_patterns = self.find_similar_patterns(current_context)
            
            # Get related decisions
            related_decisions = self.find_related_decisions(current_context)
            
            # Get project evolution context
            project_context = self.get_project_evolution_context()
            
            return {
                'patterns': relevant_patterns,
                'decisions': related_decisions,
                'project_context': project_context,
                'retrieved_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get relevant knowledge: {e}")
            return {'error': str(e)}
    
    def find_similar_patterns(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar patterns from knowledge base."""
        try:
            # Get recent high-frequency patterns
            patterns = self.knowledge_db.execute("""
                SELECT * FROM knowledge_patterns 
                WHERE frequency > 1 
                ORDER BY frequency DESC, last_seen DESC 
                LIMIT 10
            """).fetchall()
            
            return [dict(row) for row in patterns]
            
        except Exception as e:
            logging.error(f"Failed to find similar patterns: {e}")
            return []
    
    def find_related_decisions(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find related decisions from knowledge base."""
        try:
            # Get recent high-impact decisions
            decisions = self.knowledge_db.execute("""
                SELECT * FROM project_decisions 
                WHERE impact_score > 0.5 
                ORDER BY impact_score DESC, created_at DESC 
                LIMIT 5
            """).fetchall()
            
            return [dict(row) for row in decisions]
            
        except Exception as e:
            logging.error(f"Failed to find related decisions: {e}")
            return []
    
    def get_project_evolution_context(self) -> Dict[str, Any]:
        """Get project evolution context."""
        try:
            # Get project memory insights
            insights = self.knowledge_db.execute("""
                SELECT * FROM project_memory 
                WHERE memory_type = 'session_insight' 
                ORDER BY importance_score DESC, created_at DESC 
                LIMIT 10
            """).fetchall()
            
            return {
                'recent_insights': [dict(row) for row in insights],
                'total_patterns': self.knowledge_db.execute("SELECT COUNT(*) FROM knowledge_patterns").fetchone()[0],
                'total_decisions': self.knowledge_db.execute("SELECT COUNT(*) FROM project_decisions").fetchone()[0],
                'knowledge_age_days': self._calculate_knowledge_age()
            }
            
        except Exception as e:
            logging.error(f"Failed to get project evolution context: {e}")
            return {}
    
    def _calculate_knowledge_age(self) -> int:
        """Calculate age of knowledge base in days."""
        try:
            oldest = self.knowledge_db.execute("""
                SELECT MIN(created_at) FROM (
                    SELECT created_at FROM knowledge_patterns
                    UNION ALL
                    SELECT created_at FROM project_decisions
                    UNION ALL
                    SELECT created_at FROM project_memory
                )
            """).fetchone()
            
            if oldest and oldest[0]:
                oldest_date = datetime.fromisoformat(oldest[0])
                return (datetime.now() - oldest_date).days
            
            return 0
            
        except Exception as e:
            logging.error(f"Failed to calculate knowledge age: {e}")
            return 0
    
    def cleanup_old_knowledge(self, retention_days: int = 90):
        """Clean up old knowledge entries."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
            
            # Clean up old patterns with low frequency
            self.knowledge_db.execute("""
                DELETE FROM knowledge_patterns 
                WHERE created_at < ? AND frequency = 1
            """, (cutoff_date,))
            
            # Clean up old low-importance memory entries
            self.knowledge_db.execute("""
                DELETE FROM project_memory 
                WHERE created_at < ? AND importance_score < 0.3
            """, (cutoff_date,))
            
            self.knowledge_db.commit()
            logging.info(f"Cleaned up knowledge older than {retention_days} days")
            
        except Exception as e:
            logging.error(f"Failed to cleanup old knowledge: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            stats = {}
            
            # Pattern statistics
            pattern_stats = self.knowledge_db.execute("""
                SELECT pattern_type, COUNT(*) as count, AVG(frequency) as avg_frequency
                FROM knowledge_patterns 
                GROUP BY pattern_type
            """).fetchall()
            
            stats['patterns'] = {row['pattern_type']: {'count': row['count'], 'avg_frequency': row['avg_frequency']} 
                               for row in pattern_stats}
            
            # Decision statistics
            decision_stats = self.knowledge_db.execute("""
                SELECT decision_type, COUNT(*) as count, AVG(impact_score) as avg_impact
                FROM project_decisions 
                GROUP BY decision_type
            """).fetchall()
            
            stats['decisions'] = {row['decision_type']: {'count': row['count'], 'avg_impact': row['avg_impact']} 
                                for row in decision_stats}
            
            # General statistics
            stats['totals'] = {
                'patterns': self.knowledge_db.execute("SELECT COUNT(*) FROM knowledge_patterns").fetchone()[0],
                'decisions': self.knowledge_db.execute("SELECT COUNT(*) FROM project_decisions").fetchone()[0],
                'memory_entries': self.knowledge_db.execute("SELECT COUNT(*) FROM project_memory").fetchone()[0],
                'relationships': self.knowledge_db.execute("SELECT COUNT(*) FROM session_relationships").fetchone()[0]
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Failed to get knowledge stats: {e}")
            return {'error': str(e)}