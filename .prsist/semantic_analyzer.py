#!/usr/bin/env python3
"""
Semantic Analyzer for Prsist Memory System - Phase 3.
Provides code semantic analysis, similarity scoring, and intelligent context correlation.
"""

import os
import re
import ast
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import sqlite3

# Optional imports for advanced semantic analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available - using basic similarity calculations")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - using basic text similarity")

from database import MemoryDatabase


@dataclass
class CodeElement:
    """Represents a semantic code element."""
    element_type: str  # function, class, variable, import, etc.
    name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    dependencies: List[str] = None
    complexity_score: float = 0.0
    semantic_hash: str = ""


@dataclass
class SemanticEmbedding:
    """Represents a semantic embedding of code or context."""
    content_id: str
    content_type: str  # file, function, session, commit
    embedding_vector: List[float]
    metadata: Dict[str, Any]
    created_at: datetime


class SemanticAnalyzer:
    """Analyzes code semantics and generates embeddings for similarity matching."""
    
    def __init__(self, memory_dir: str, repo_path: str = "."):
        """Initialize semantic analyzer."""
        self.memory_dir = Path(memory_dir)
        self.repo_path = Path(repo_path).resolve()
        
        # Initialize components
        self.db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
        
        # Cache for embeddings and analysis
        self._embedding_cache = {}
        self._code_element_cache = {}
        
        # TF-IDF vectorizer for text similarity
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
        else:
            self.tfidf_vectorizer = None
        
        logging.info("Semantic Analyzer initialized")
    
    def analyze_file_semantics(self, file_path: str) -> Dict[str, Any]:
        """Analyze semantic structure of a code file."""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return {"error": "File not found", "file_path": file_path}
            
            # Check cache first
            file_hash = self._get_file_hash(file_path)
            cache_key = f"file_semantics_{file_hash}"
            
            if cache_key in self._code_element_cache:
                return self._code_element_cache[cache_key]
            
            # Read file content
            content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
            file_extension = file_path_obj.suffix.lower()
            
            # Analyze based on file type
            if file_extension == '.py':
                analysis = self._analyze_python_file(content, file_path)
            elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
                analysis = self._analyze_javascript_file(content, file_path)
            elif file_extension in ['.java']:
                analysis = self._analyze_java_file(content, file_path)
            elif file_extension in ['.cpp', '.c', '.h', '.hpp']:
                analysis = self._analyze_cpp_file(content, file_path)
            else:
                analysis = self._analyze_generic_file(content, file_path)
            
            # Add general metadata
            analysis.update({
                "file_path": file_path,
                "file_hash": file_hash,
                "file_size": len(content),
                "line_count": len(content.splitlines()),
                "analysis_timestamp": datetime.now().isoformat(),
                "semantic_complexity": self._calculate_file_complexity(analysis)
            })
            
            # Generate semantic embedding
            if self.tfidf_vectorizer:
                embedding = self._generate_text_embedding(content, file_path, "file")
                analysis["embedding_id"] = embedding.content_id if embedding else None
            
            # Cache the result
            self._code_element_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze file semantics for {file_path}: {e}")
            return {"error": str(e), "file_path": file_path}
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content for caching."""
        try:
            content = Path(file_path).read_bytes()
            return hashlib.md5(content).hexdigest()[:16]
        except:
            return "unknown"
    
    def _analyze_python_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Python file semantics."""
        elements = []
        imports = []
        complexity_score = 0
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    element = self._extract_python_function(node, content, file_path)
                    elements.append(element)
                    complexity_score += element.complexity_score
                
                elif isinstance(node, ast.ClassDef):
                    element = self._extract_python_class(node, content, file_path)
                    elements.append(element)
                    complexity_score += element.complexity_score
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._extract_python_import(node)
                    imports.extend(import_info)
        
        except SyntaxError as e:
            logging.warning(f"Python syntax error in {file_path}: {e}")
            return self._analyze_generic_file(content, file_path)
        
        return {
            "language": "python",
            "elements": [self._element_to_dict(e) for e in elements],
            "imports": imports,
            "function_count": sum(1 for e in elements if e.element_type == "function"),
            "class_count": sum(1 for e in elements if e.element_type == "class"),
            "total_complexity": complexity_score,
            "semantic_keywords": self._extract_semantic_keywords(content)
        }
    
    def _extract_python_function(self, node: ast.FunctionDef, content: str, file_path: str) -> CodeElement:
        """Extract Python function information."""
        lines = content.splitlines()
        
        # Get function signature
        signature_parts = [node.name, "("]
        for arg in node.args.args:
            signature_parts.append(arg.arg)
            if arg.annotation:
                signature_parts.append(f": {ast.unparse(arg.annotation)}")
            signature_parts.append(", ")
        if signature_parts[-1] == ", ":
            signature_parts.pop()
        signature_parts.append(")")
        
        if node.returns:
            signature_parts.extend([" -> ", ast.unparse(node.returns)])
        
        signature = "".join(signature_parts)
        
        # Get docstring
        docstring = ""
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant) 
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Calculate complexity (cyclomatic complexity approximation)
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return CodeElement(
            element_type="function",
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=signature,
            docstring=docstring,
            complexity_score=complexity,
            semantic_hash=hashlib.md5(signature.encode()).hexdigest()[:16]
        )
    
    def _extract_python_class(self, node: ast.ClassDef, content: str, file_path: str) -> CodeElement:
        """Extract Python class information."""
        # Get base classes
        bases = [ast.unparse(base) for base in node.bases] if node.bases else []
        signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
        
        # Get class docstring
        docstring = ""
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant) 
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Calculate complexity based on methods and nested structures
        complexity = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        
        return CodeElement(
            element_type="class",
            name=node.name,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            signature=signature,
            docstring=docstring,
            dependencies=bases,
            complexity_score=complexity,
            semantic_hash=hashlib.md5(signature.encode()).hexdigest()[:16]
        )
    
    def _extract_python_import(self, node: ast.Import | ast.ImportFrom) -> List[Dict[str, str]]:
        """Extract Python import information."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "alias": alias.asname,
                    "type": "import"
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "module": f"{module}.{alias.name}" if module else alias.name,
                    "alias": alias.asname,
                    "from_module": module,
                    "type": "from_import"
                })
        
        return imports
    
    def _analyze_javascript_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file semantics."""
        elements = []
        imports = []
        
        # Basic regex-based analysis (would be better with proper parser)
        # Function declarations
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))'
        for match in re.finditer(function_pattern, content):
            name = match.group(1) or match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            elements.append({
                "type": "function",
                "name": name,
                "line": line_num,
                "signature": match.group(0)
            })
        
        # Class declarations
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            extends = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            elements.append({
                "type": "class",
                "name": name,
                "line": line_num,
                "extends": extends,
                "signature": match.group(0)
            })
        
        # Import statements
        import_pattern = r'import\s+(?:{([^}]+)}|\*\s+as\s+(\w+)|([^"\']+))\s+from\s+["\']([^"\']+)["\']'
        for match in re.finditer(import_pattern, content):
            imports.append({
                "imported": match.group(1) or match.group(2) or match.group(3),
                "from": match.group(4),
                "type": "es6_import"
            })
        
        return {
            "language": "javascript",
            "elements": elements,
            "imports": imports,
            "function_count": sum(1 for e in elements if e["type"] == "function"),
            "class_count": sum(1 for e in elements if e["type"] == "class"),
            "semantic_keywords": self._extract_semantic_keywords(content)
        }
    
    def _analyze_java_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Java file semantics."""
        elements = []
        imports = []
        
        # Package declaration
        package_match = re.search(r'package\s+([\w.]+);', content)
        package = package_match.group(1) if package_match else None
        
        # Import statements
        import_pattern = r'import\s+(?:static\s+)?([\w.*]+);'
        for match in re.finditer(import_pattern, content):
            imports.append({
                "module": match.group(1),
                "type": "java_import"
            })
        
        # Class/interface declarations
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            elements.append({
                "type": "class",
                "name": name,
                "line": line_num,
                "signature": match.group(0)
            })
        
        # Method declarations
        method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(method_pattern, content):
            name = match.group(1)
            if name not in ['if', 'for', 'while', 'switch']:  # Filter out keywords
                line_num = content[:match.start()].count('\n') + 1
                elements.append({
                    "type": "method",
                    "name": name,
                    "line": line_num,
                    "signature": match.group(0)
                })
        
        return {
            "language": "java",
            "package": package,
            "elements": elements,
            "imports": imports,
            "class_count": sum(1 for e in elements if e["type"] == "class"),
            "method_count": sum(1 for e in elements if e["type"] == "method"),
            "semantic_keywords": self._extract_semantic_keywords(content)
        }
    
    def _analyze_cpp_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze C++ file semantics."""
        elements = []
        includes = []
        
        # Include statements
        include_pattern = r'#include\s+[<"](.*?)[>"]'
        for match in re.finditer(include_pattern, content):
            includes.append({
                "file": match.group(1),
                "type": "cpp_include"
            })
        
        # Namespace declarations
        namespace_pattern = r'namespace\s+(\w+)'
        namespaces = [match.group(1) for match in re.finditer(namespace_pattern, content)]
        
        # Class/struct declarations
        class_pattern = r'(?:class|struct)\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            elements.append({
                "type": "class",
                "name": name,
                "line": line_num,
                "signature": match.group(0)
            })
        
        # Function declarations
        function_pattern = r'(?:(?:inline|static|virtual|explicit)\s+)*[\w:*&<>]+\s+(\w+)\s*\([^)]*\)(?:\s*const)?(?:\s*override)?'
        for match in re.finditer(function_pattern, content):
            name = match.group(1)
            if name not in ['if', 'for', 'while', 'switch', 'return']:  # Filter keywords
                line_num = content[:match.start()].count('\n') + 1
                elements.append({
                    "type": "function",
                    "name": name,
                    "line": line_num,
                    "signature": match.group(0)
                })
        
        return {
            "language": "cpp",
            "elements": elements,
            "includes": includes,
            "namespaces": namespaces,
            "class_count": sum(1 for e in elements if e["type"] == "class"),
            "function_count": sum(1 for e in elements if e["type"] == "function"),
            "semantic_keywords": self._extract_semantic_keywords(content)
        }
    
    def _analyze_generic_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze generic file semantics."""
        return {
            "language": "unknown",
            "elements": [],
            "imports": [],
            "semantic_keywords": self._extract_semantic_keywords(content),
            "word_count": len(content.split()),
            "identifier_pattern": self._extract_identifiers(content)
        }
    
    def _extract_semantic_keywords(self, content: str) -> List[str]:
        """Extract semantic keywords from content."""
        # Common programming keywords and domain-specific terms
        programming_keywords = {
            'function', 'class', 'method', 'variable', 'import', 'export', 'module',
            'interface', 'abstract', 'extends', 'implements', 'inherit', 'override',
            'public', 'private', 'protected', 'static', 'final', 'const', 'let', 'var',
            'async', 'await', 'promise', 'callback', 'event', 'handler', 'listener',
            'database', 'query', 'model', 'view', 'controller', 'service', 'component',
            'api', 'endpoint', 'request', 'response', 'middleware', 'authentication',
            'validation', 'error', 'exception', 'logging', 'debug', 'test', 'mock'
        }
        
        # Extract words that might be semantic keywords
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content.lower())
        word_counts = Counter(words)
        
        # Return frequent words that might be semantically meaningful
        semantic_keywords = []
        for word, count in word_counts.most_common(50):
            if (len(word) > 3 and count > 1 and 
                (word in programming_keywords or word.endswith(('er', 'or', 'ing', 'ed')))):
                semantic_keywords.append(word)
        
        return semantic_keywords[:20]  # Top 20 semantic keywords
    
    def _extract_identifiers(self, content: str) -> Dict[str, int]:
        """Extract identifier patterns from content."""
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content)
        
        patterns = {
            "camelCase": 0,
            "snake_case": 0,
            "PascalCase": 0,
            "UPPER_CASE": 0,
            "kebab-case": 0
        }
        
        for identifier in identifiers:
            if re.match(r'^[a-z][a-zA-Z0-9]*$', identifier):
                patterns["camelCase"] += 1
            elif re.match(r'^[a-z][a-z0-9_]*$', identifier):
                patterns["snake_case"] += 1
            elif re.match(r'^[A-Z][a-zA-Z0-9]*$', identifier):
                patterns["PascalCase"] += 1
            elif re.match(r'^[A-Z][A-Z0-9_]*$', identifier):
                patterns["UPPER_CASE"] += 1
            elif '-' in identifier:
                patterns["kebab-case"] += 1
        
        return patterns
    
    def _calculate_file_complexity(self, analysis: Dict) -> float:
        """Calculate overall file complexity score."""
        elements = analysis.get("elements", [])
        if not elements:
            return 0.0
        
        total_complexity = 0
        for element in elements:
            if isinstance(element, dict):
                total_complexity += element.get("complexity_score", 1)
            else:
                total_complexity += getattr(element, 'complexity_score', 1)
        
        # Normalize by number of elements
        base_complexity = total_complexity / len(elements) if elements else 0
        
        # Adjust for file size and imports
        size_factor = min(analysis.get("line_count", 0) / 100, 2.0)
        import_factor = min(len(analysis.get("imports", [])) / 10, 1.5)
        
        return round(base_complexity * (1 + size_factor * 0.1 + import_factor * 0.1), 2)
    
    def _element_to_dict(self, element: CodeElement) -> Dict[str, Any]:
        """Convert CodeElement to dictionary."""
        return {
            "type": element.element_type,
            "name": element.name,
            "file_path": element.file_path,
            "line_start": element.line_start,
            "line_end": element.line_end,
            "signature": element.signature,
            "docstring": element.docstring,
            "dependencies": element.dependencies or [],
            "complexity_score": element.complexity_score,
            "semantic_hash": element.semantic_hash
        }
    
    def generate_session_embedding(self, session_id: str) -> Optional[SemanticEmbedding]:
        """Generate semantic embedding for a session."""
        try:
            # Get session data
            session = self.db.get_session(session_id)
            if not session:
                return None
            
            # Collect session content
            content_parts = []
            
            # Add context data
            context_data = session.get('context_data', {})
            if isinstance(context_data, dict):
                content_parts.extend([
                    str(context_data.get('current_task', '')),
                    str(context_data.get('project_context', '')),
                    str(context_data.get('working_directory', ''))
                ])
            
            # Add tool usage information
            tools = self.db.get_session_tool_usage(session_id)
            tool_names = [tool.get('tool_name', '') for tool in tools]
            content_parts.append(' '.join(tool_names))
            
            # Add file interactions
            file_interactions = []
            for tool in tools:
                tool_input = tool.get('input_data')
                if isinstance(tool_input, dict) and 'file_path' in tool_input:
                    file_interactions.append(tool_input['file_path'])
            
            content_parts.append(' '.join(file_interactions))
            
            # Combine content
            combined_content = ' '.join(filter(None, content_parts))
            
            if not combined_content.strip():
                return None
            
            # Generate embedding
            return self._generate_text_embedding(combined_content, session_id, "session")
            
        except Exception as e:
            logging.error(f"Failed to generate session embedding for {session_id}: {e}")
            return None
    
    def generate_commit_embedding(self, commit_sha: str) -> Optional[SemanticEmbedding]:
        """Generate semantic embedding for a commit."""
        try:
            # Get commit data
            commit = self.db.get_commit_by_sha(commit_sha)
            if not commit:
                return None
            
            # Collect commit content
            content_parts = [
                commit.get('commit_message', ''),
                commit.get('branch_name', ''),
                commit.get('author_email', '')
            ]
            
            # Add commit metadata
            metadata = commit.get('commit_metadata', {})
            if isinstance(metadata, dict):
                content_parts.extend([
                    str(metadata.get('commit_type', '')),
                    ' '.join(metadata.get('branches', []))
                ])
            
            # Add file change information
            # Note: This would be enhanced with actual diff content
            files_changed = commit.get('changed_files_count', 0)
            content_parts.append(f"files_changed_{files_changed}")
            
            # Combine content
            combined_content = ' '.join(filter(None, content_parts))
            
            if not combined_content.strip():
                return None
            
            return self._generate_text_embedding(combined_content, commit_sha, "commit")
            
        except Exception as e:
            logging.error(f"Failed to generate commit embedding for {commit_sha}: {e}")
            return None
    
    def _generate_text_embedding(self, text: str, content_id: str, content_type: str) -> Optional[SemanticEmbedding]:
        """Generate text embedding using available methods."""
        try:
            if not text or not text.strip():
                return None
            
            # Use TF-IDF if available
            if self.tfidf_vectorizer and SKLEARN_AVAILABLE:
                # Fit on single document (not ideal, but works for basic similarity)
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
                embedding_vector = tfidf_matrix.toarray()[0].tolist()
            else:
                # Fallback to simple word frequency embedding
                embedding_vector = self._generate_simple_embedding(text)
            
            embedding = SemanticEmbedding(
                content_id=content_id,
                content_type=content_type,
                embedding_vector=embedding_vector,
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "generated_method": "tfidf" if self.tfidf_vectorizer else "simple"
                },
                created_at=datetime.now()
            )
            
            # Store embedding in cache
            self._embedding_cache[f"{content_type}_{content_id}"] = embedding
            
            return embedding
            
        except Exception as e:
            logging.error(f"Failed to generate text embedding: {e}")
            return None
    
    def _generate_simple_embedding(self, text: str, dimension: int = 100) -> List[float]:
        """Generate simple embedding based on word frequencies."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        
        # Create a simple hash-based embedding
        embedding = [0.0] * dimension
        
        for word, count in word_counts.most_common(50):
            # Use hash to distribute words across dimensions
            hash_val = hash(word) % dimension
            embedding[hash_val] += count / len(words)
        
        # Normalize
        total = sum(embedding)
        if total > 0:
            embedding = [v / total for v in embedding]
        
        return embedding
    
    def calculate_similarity(self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding) -> float:
        """Calculate similarity between two embeddings."""
        try:
            if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
                # Use cosine similarity
                vec1 = np.array(embedding1.embedding_vector).reshape(1, -1)
                vec2 = np.array(embedding2.embedding_vector).reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                return float(similarity)
            else:
                # Simple dot product similarity
                vec1 = embedding1.embedding_vector
                vec2 = embedding2.embedding_vector
                
                if len(vec1) != len(vec2):
                    return 0.0
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5
                
                if magnitude1 == 0 or magnitude2 == 0:
                    return 0.0
                
                return dot_product / (magnitude1 * magnitude2)
                
        except Exception as e:
            logging.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def find_similar_sessions(self, session_id: str, min_similarity: float = 0.3, limit: int = 10) -> List[Dict[str, Any]]:
        """Find sessions similar to the given session."""
        try:
            # Generate embedding for target session
            target_embedding = self.generate_session_embedding(session_id)
            if not target_embedding:
                return []
            
            # Get all sessions to compare against
            all_sessions = self.db.get_recent_sessions(limit=100)
            similar_sessions = []
            
            for session in all_sessions:
                if session['id'] == session_id:
                    continue
                
                # Generate embedding for comparison session
                comparison_embedding = self.generate_session_embedding(session['id'])
                if not comparison_embedding:
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(target_embedding, comparison_embedding)
                
                if similarity >= min_similarity:
                    similar_sessions.append({
                        "session_id": session['id'],
                        "similarity_score": round(similarity, 3),
                        "session_data": session,
                        "created_at": session.get('created_at'),
                        "context_overlap": self._analyze_context_overlap(target_embedding, comparison_embedding)
                    })
            
            # Sort by similarity score
            similar_sessions.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_sessions[:limit]
            
        except Exception as e:
            logging.error(f"Failed to find similar sessions: {e}")
            return []
    
    def find_similar_commits(self, commit_sha: str, min_similarity: float = 0.3, limit: int = 10) -> List[Dict[str, Any]]:
        """Find commits similar to the given commit."""
        try:
            # Generate embedding for target commit
            target_embedding = self.generate_commit_embedding(commit_sha)
            if not target_embedding:
                return []
            
            # Get all commits to compare against
            all_commits = self.db.get_recent_commits(limit=100)
            similar_commits = []
            
            for commit in all_commits:
                if commit['commit_sha'] == commit_sha:
                    continue
                
                # Generate embedding for comparison commit
                comparison_embedding = self.generate_commit_embedding(commit['commit_sha'])
                if not comparison_embedding:
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(target_embedding, comparison_embedding)
                
                if similarity >= min_similarity:
                    similar_commits.append({
                        "commit_sha": commit['commit_sha'],
                        "similarity_score": round(similarity, 3),
                        "commit_data": commit,
                        "commit_message": commit.get('commit_message'),
                        "branch_name": commit.get('branch_name')
                    })
            
            # Sort by similarity score
            similar_commits.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_commits[:limit]
            
        except Exception as e:
            logging.error(f"Failed to find similar commits: {e}")
            return []
    
    def _analyze_context_overlap(self, embedding1: SemanticEmbedding, embedding2: SemanticEmbedding) -> Dict[str, Any]:
        """Analyze overlap between embeddings."""
        return {
            "content_types": [embedding1.content_type, embedding2.content_type],
            "metadata_similarity": self._compare_metadata(embedding1.metadata, embedding2.metadata),
            "embedding_dimensions": len(embedding1.embedding_vector)
        }
    
    def _compare_metadata(self, meta1: Dict, meta2: Dict) -> float:
        """Compare metadata similarity."""
        common_keys = set(meta1.keys()) & set(meta2.keys())
        if not common_keys:
            return 0.0
        
        similar_values = 0
        for key in common_keys:
            if meta1[key] == meta2[key]:
                similar_values += 1
        
        return similar_values / len(common_keys)
    
    def analyze_code_evolution(self, file_path: str, time_period_days: int = 30) -> Dict[str, Any]:
        """Analyze how code has evolved over time."""
        try:
            # Get commits that modified this file
            commits = []
            all_commits = self.db.get_recent_commits(limit=200)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=time_period_days)
            
            for commit in all_commits:
                try:
                    commit_time = datetime.fromisoformat(commit.get('commit_timestamp', ''))
                    if start_time <= commit_time <= end_time:
                        commits.append(commit)
                except:
                    continue
            
            # Analyze evolution patterns
            evolution = {
                "file_path": file_path,
                "analysis_period_days": time_period_days,
                "total_commits": len(commits),
                "commit_timeline": self._build_commit_timeline(commits),
                "complexity_evolution": self._analyze_complexity_evolution(file_path, commits),
                "semantic_changes": self._analyze_semantic_changes(file_path, commits)
            }
            
            return evolution
            
        except Exception as e:
            logging.error(f"Failed to analyze code evolution: {e}")
            return {"error": str(e)}
    
    def _build_commit_timeline(self, commits: List[Dict]) -> List[Dict[str, Any]]:
        """Build timeline of commits."""
        timeline = []
        for commit in sorted(commits, key=lambda c: c.get('commit_timestamp', '')):
            timeline.append({
                "commit_sha": commit['commit_sha'][:8],
                "timestamp": commit.get('commit_timestamp'),
                "message": commit.get('commit_message', '')[:60],
                "lines_added": commit.get('lines_added', 0),
                "lines_deleted": commit.get('lines_deleted', 0),
                "files_changed": commit.get('changed_files_count', 0)
            })
        return timeline
    
    def _analyze_complexity_evolution(self, file_path: str, commits: List[Dict]) -> Dict[str, Any]:
        """Analyze complexity evolution over commits."""
        # This would ideally analyze the file at each commit
        # For now, return basic metrics
        if not commits:
            return {"trend": "no_data"}
        
        lines_over_time = []
        for commit in sorted(commits, key=lambda c: c.get('commit_timestamp', '')):
            net_lines = commit.get('lines_added', 0) - commit.get('lines_deleted', 0)
            lines_over_time.append(net_lines)
        
        # Simple trend analysis
        if len(lines_over_time) >= 3:
            first_half = lines_over_time[:len(lines_over_time)//2]
            second_half = lines_over_time[len(lines_over_time)//2:]
            
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            
            if second_avg > first_avg * 1.2:
                trend = "increasing"
            elif second_avg < first_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "total_line_changes": sum(lines_over_time),
            "avg_change_per_commit": round(sum(lines_over_time) / len(lines_over_time), 2) if lines_over_time else 0
        }
    
    def _analyze_semantic_changes(self, file_path: str, commits: List[Dict]) -> Dict[str, Any]:
        """Analyze semantic changes over time."""
        # Basic analysis - would be enhanced with actual file content at each commit
        commit_types = Counter()
        for commit in commits:
            metadata = commit.get('commit_metadata', {})
            commit_type = metadata.get('commit_type', 'unknown')
            commit_types[commit_type] += 1
        
        return {
            "change_types": dict(commit_types),
            "primary_change_type": commit_types.most_common(1)[0][0] if commit_types else "unknown",
            "change_diversity": len(commit_types)
        }


# CLI interface for semantic analysis
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: semantic_analyzer.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    memory_dir = os.environ.get("PRSIST_MEMORY_DIR", os.path.dirname(__file__))
    
    analyzer = SemanticAnalyzer(memory_dir)
    
    if command == "analyze_file":
        file_path = sys.argv[2] if len(sys.argv) > 2 else ""
        if file_path:
            result = analyzer.analyze_file_semantics(file_path)
            print(json.dumps(result, indent=2))
        else:
            print("Error: file_path required")
            sys.exit(1)
    
    elif command == "similar_sessions":
        session_id = sys.argv[2] if len(sys.argv) > 2 else ""
        min_similarity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        if session_id:
            result = analyzer.find_similar_sessions(session_id, min_similarity)
            print(json.dumps(result, indent=2))
        else:
            print("Error: session_id required")
            sys.exit(1)
    
    elif command == "similar_commits":
        commit_sha = sys.argv[2] if len(sys.argv) > 2 else ""
        min_similarity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
        if commit_sha:
            result = analyzer.find_similar_commits(commit_sha, min_similarity)
            print(json.dumps(result, indent=2))
        else:
            print("Error: commit_sha required")
            sys.exit(1)
    
    elif command == "code_evolution":
        file_path = sys.argv[2] if len(sys.argv) > 2 else ""
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        if file_path:
            result = analyzer.analyze_code_evolution(file_path, days)
            print(json.dumps(result, indent=2))
        else:
            print("Error: file_path required")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)