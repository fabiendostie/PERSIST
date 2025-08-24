#!/usr/bin/env python3
"""
Memory relevance scoring system for Prsist Memory System Phase 3.
Multi-dimensional relevance scoring with embeddings support.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import re

# Try to import sentence transformers, fall back if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

from utils import setup_logging

class RelevanceScorer:
    """Multi-dimensional relevance scoring for memory entries."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize relevance scorer."""
        self.weights = {
            'recency': 0.3,
            'importance': 0.25,
            'similarity': 0.25,
            'role_relevance': 0.2
        }
        
        # Initialize embedding model if available
        self.embedding_model = None
        self.embeddings_available = EMBEDDINGS_AVAILABLE
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logging.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load embedding model: {e}")
                self.embeddings_available = False
        else:
            logging.warning("Sentence transformers not available, using fallback similarity")
    
    def calculate_relevance(self, memory_entry: Dict[str, Any], 
                          current_context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate multi-dimensional relevance score."""
        try:
            scores = {}
            
            # Recency score (time-based decay)
            scores['recency'] = self.calculate_recency_score(
                memory_entry.get('timestamp', datetime.now().isoformat())
            )
            
            # Importance score (stored or calculated)
            scores['importance'] = self.calculate_importance_score(memory_entry)
            
            # Semantic similarity score
            scores['similarity'] = self.calculate_similarity_score(
                memory_entry.get('content', ''),
                current_context.get('current_task', '')
            )
            
            # Role-specific relevance
            scores['role_relevance'] = self.calculate_role_relevance(
                memory_entry,
                current_context.get('active_agent', 'general')
            )
            
            # Weighted final score
            final_score = sum(
                scores[dimension] * self.weights[dimension]
                for dimension in scores
            )
            
            return min(final_score, 1.0), scores
            
        except Exception as e:
            logging.error(f"Failed to calculate relevance: {e}")
            return 0.0, {}
    
    def calculate_recency_score(self, timestamp: str) -> float:
        """Calculate exponential decay based on time."""
        try:
            if isinstance(timestamp, str):
                entry_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                entry_time = timestamp
            
            now = datetime.now()
            if entry_time.tzinfo is not None:
                # Make both timezone-aware
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
            
            age_hours = (now - entry_time).total_seconds() / 3600
            
            # Decay function: score = e^(-age/24) for 24-hour half-life
            decay_rate = 0.029  # ln(2)/24
            score = np.exp(-decay_rate * age_hours)
            
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            logging.error(f"Failed to calculate recency score: {e}")
            return 0.5  # Default moderate recency
    
    def calculate_importance_score(self, memory_entry: Dict[str, Any]) -> float:
        """Calculate importance score for memory entry."""
        try:
            # Use stored importance if available
            stored_importance = memory_entry.get('importance_score')
            if stored_importance is not None:
                return max(0.0, min(float(stored_importance), 1.0))
            
            # Calculate importance based on entry characteristics
            importance = 0.5  # Default baseline
            
            # Boost for certain entry types
            entry_type = memory_entry.get('type', '')
            type_boosts = {
                'decision': 0.8,
                'critical_error': 0.9,
                'breakthrough': 0.8,
                'milestone': 0.7,
                'configuration': 0.6,
                'pattern': 0.6,
                'lesson_learned': 0.7
            }
            
            if entry_type in type_boosts:
                importance = type_boosts[entry_type]
            
            # Boost for entries with many interactions
            interaction_count = memory_entry.get('interaction_count', 0)
            if interaction_count > 0:
                # Logarithmic boost for interactions
                interaction_boost = min(0.3, 0.1 * np.log(1 + interaction_count))
                importance += interaction_boost
            
            # Boost for entries marked as critical
            if memory_entry.get('is_critical', False):
                importance += 0.2
            
            # Boost for entries with high user rating
            user_rating = memory_entry.get('user_rating', 0)
            if user_rating > 0:
                importance += (user_rating / 5.0) * 0.2  # Assuming 1-5 rating scale
            
            return max(0.0, min(importance, 1.0))
            
        except Exception as e:
            logging.error(f"Failed to calculate importance score: {e}")
            return 0.5
    
    def calculate_similarity_score(self, memory_content: str, current_task: str) -> float:
        """Calculate semantic similarity using embeddings or fallback."""
        try:
            if not memory_content or not current_task:
                return 0.0
            
            if self.embeddings_available and self.embedding_model:
                return self._calculate_embedding_similarity(memory_content, current_task)
            else:
                return self._calculate_fallback_similarity(memory_content, current_task)
                
        except Exception as e:
            logging.error(f"Failed to calculate similarity score: {e}")
            return 0.0
    
    def _calculate_embedding_similarity(self, memory_content: str, current_task: str) -> float:
        """Calculate similarity using sentence embeddings."""
        try:
            # Encode both texts
            memory_embedding = self.embedding_model.encode([memory_content])
            task_embedding = self.embedding_model.encode([current_task])
            
            # Calculate cosine similarity
            similarity = np.dot(memory_embedding[0], task_embedding[0]) / (
                np.linalg.norm(memory_embedding[0]) * 
                np.linalg.norm(task_embedding[0])
            )
            
            # Ensure non-negative and within bounds
            return max(0.0, min(similarity, 1.0))
            
        except Exception as e:
            logging.error(f"Failed to calculate embedding similarity: {e}")
            return self._calculate_fallback_similarity(memory_content, current_task)
    
    def _calculate_fallback_similarity(self, memory_content: str, current_task: str) -> float:
        """Calculate similarity using keyword matching and heuristics."""
        try:
            # Clean and tokenize texts
            memory_words = set(re.findall(r'\w+', memory_content.lower()))
            task_words = set(re.findall(r'\w+', current_task.lower()))
            
            if not memory_words or not task_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(memory_words.intersection(task_words))
            union = len(memory_words.union(task_words))
            
            jaccard = intersection / union if union > 0 else 0.0
            
            # Boost for exact phrase matches
            memory_phrases = self._extract_phrases(memory_content.lower())
            task_phrases = self._extract_phrases(current_task.lower())
            
            phrase_matches = len(set(memory_phrases).intersection(set(task_phrases)))
            phrase_boost = min(0.3, phrase_matches * 0.1)
            
            # Boost for important keyword matches
            important_words = {
                'implement', 'create', 'build', 'develop', 'fix', 'debug', 
                'test', 'optimize', 'refactor', 'integrate', 'configure'
            }
            
            important_matches = len(
                memory_words.intersection(task_words).intersection(important_words)
            )
            importance_boost = min(0.2, important_matches * 0.05)
            
            total_similarity = jaccard + phrase_boost + importance_boost
            return max(0.0, min(total_similarity, 1.0))
            
        except Exception as e:
            logging.error(f"Failed to calculate fallback similarity: {e}")
            return 0.0
    
    def _extract_phrases(self, text: str, min_length: int = 2) -> List[str]:
        """Extract meaningful phrases from text."""
        try:
            # Simple phrase extraction using common patterns
            phrases = []
            
            # Extract quoted phrases
            quoted = re.findall(r'"([^"]*)"', text)
            phrases.extend(quoted)
            
            # Extract camelCase and snake_case identifiers
            identifiers = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b', text)
            phrases.extend(identifiers)
            
            # Extract technical terms (words with specific patterns)
            tech_terms = re.findall(r'\b[a-z]*(?:config|manager|system|engine|processor)\b', text)
            phrases.extend(tech_terms)
            
            # Filter by minimum length
            return [p for p in phrases if len(p) >= min_length]
            
        except Exception as e:
            logging.error(f"Failed to extract phrases: {e}")
            return []
    
    def calculate_role_relevance(self, memory_entry: Dict[str, Any], 
                               active_agent: str) -> float:
        """Calculate role-specific relevance."""
        try:
            # Default relevance
            relevance = 0.5
            
            # Check if memory entry has role-specific information
            entry_roles = memory_entry.get('relevant_roles', [])
            if not entry_roles:
                entry_roles = self._infer_roles_from_content(memory_entry)
            
            # Calculate role match score
            if active_agent in entry_roles:
                relevance = 0.9  # High relevance for exact role match
            elif any(self._roles_are_related(active_agent, role) for role in entry_roles):
                relevance = 0.7  # Medium relevance for related roles
            elif entry_roles:
                relevance = 0.3  # Low relevance for unrelated roles
            # else: keep default 0.5 for role-agnostic content
            
            # Boost for entries created by the same agent
            entry_creator = memory_entry.get('created_by_agent', '')
            if entry_creator == active_agent:
                relevance += 0.1
            
            return max(0.0, min(relevance, 1.0))
            
        except Exception as e:
            logging.error(f"Failed to calculate role relevance: {e}")
            return 0.5
    
    def _infer_roles_from_content(self, memory_entry: Dict[str, Any]) -> List[str]:
        """Infer relevant roles from memory entry content."""
        try:
            content = memory_entry.get('content', '').lower()
            context = memory_entry.get('context', {})
            
            roles = []
            
            # Role inference patterns
            role_patterns = {
                'developer': ['code', 'implement', 'function', 'class', 'method', 'variable'],
                'architect': ['design', 'architecture', 'pattern', 'structure', 'framework'],
                'tester': ['test', 'verify', 'validate', 'spec', 'assertion', 'coverage'],
                'analyst': ['analyze', 'data', 'metrics', 'performance', 'profile'],
                'pm': ['project', 'timeline', 'milestone', 'delivery', 'scope'],
                'devops': ['deploy', 'infrastructure', 'pipeline', 'automation', 'monitoring'],
                'qa': ['quality', 'review', 'standard', 'compliance', 'audit'],
                'ux': ['user', 'interface', 'experience', 'usability', 'design']
            }
            
            for role, keywords in role_patterns.items():
                if any(keyword in content for keyword in keywords):
                    roles.append(role)
            
            # Check context for explicit role information
            if 'agent_type' in context:
                roles.append(context['agent_type'])
            
            # Remove duplicates
            return list(set(roles))
            
        except Exception as e:
            logging.error(f"Failed to infer roles from content: {e}")
            return []
    
    def _roles_are_related(self, role1: str, role2: str) -> bool:
        """Check if two roles are related."""
        try:
            # Define role relationships
            role_relationships = {
                'developer': ['architect', 'tester', 'analyst'],
                'architect': ['developer', 'pm', 'analyst'],
                'tester': ['developer', 'qa', 'analyst'],
                'analyst': ['developer', 'architect', 'tester', 'pm'],
                'pm': ['architect', 'analyst', 'qa'],
                'devops': ['developer', 'architect', 'qa'],
                'qa': ['tester', 'pm', 'devops', 'analyst'],
                'ux': ['pm', 'analyst']
            }
            
            role1_lower = role1.lower()
            role2_lower = role2.lower()
            
            # Check if roles are in each other's relationship lists
            related_to_role1 = role_relationships.get(role1_lower, [])
            related_to_role2 = role_relationships.get(role2_lower, [])
            
            return (role2_lower in related_to_role1 or 
                    role1_lower in related_to_role2)
            
        except Exception as e:
            logging.error(f"Failed to check role relationships: {e}")
            return False
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights."""
        try:
            # Validate weights
            if abs(sum(new_weights.values()) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
            
            # Update weights
            self.weights.update(new_weights)
            logging.info(f"Updated relevance scoring weights: {self.weights}")
            
        except Exception as e:
            logging.error(f"Failed to update weights: {e}")
    
    def batch_score_entries(self, memory_entries: List[Dict[str, Any]], 
                          current_context: Dict[str, Any], 
                          top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float, Dict[str, float]]]:
        """Score multiple memory entries and return sorted results."""
        try:
            scored_entries = []
            
            for entry in memory_entries:
                final_score, dimension_scores = self.calculate_relevance(entry, current_context)
                scored_entries.append((entry, final_score, dimension_scores))
            
            # Sort by relevance score (descending)
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k if specified
            if top_k is not None:
                scored_entries = scored_entries[:top_k]
            
            return scored_entries
            
        except Exception as e:
            logging.error(f"Failed to batch score entries: {e}")
            return []
    
    def explain_score(self, memory_entry: Dict[str, Any], 
                     current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed explanation of relevance score."""
        try:
            final_score, dimension_scores = self.calculate_relevance(memory_entry, current_context)
            
            explanation = {
                "final_score": final_score,
                "dimension_scores": dimension_scores,
                "weighted_contributions": {},
                "explanations": {}
            }
            
            # Calculate weighted contributions
            for dimension, score in dimension_scores.items():
                weighted_score = score * self.weights[dimension]
                explanation["weighted_contributions"][dimension] = weighted_score
            
            # Add explanations for each dimension
            explanation["explanations"]["recency"] = self._explain_recency(
                memory_entry.get('timestamp', '')
            )
            explanation["explanations"]["importance"] = self._explain_importance(memory_entry)
            explanation["explanations"]["similarity"] = self._explain_similarity(
                memory_entry.get('content', ''),
                current_context.get('current_task', '')
            )
            explanation["explanations"]["role_relevance"] = self._explain_role_relevance(
                memory_entry,
                current_context.get('active_agent', 'general')
            )
            
            return explanation
            
        except Exception as e:
            logging.error(f"Failed to explain score: {e}")
            return {"error": str(e)}
    
    def _explain_recency(self, timestamp: str) -> str:
        """Explain recency score."""
        try:
            if not timestamp:
                return "No timestamp available"
            
            entry_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.now() - entry_time
            
            if age.days == 0:
                return f"Very recent (today, {age.seconds // 3600} hours ago)"
            elif age.days == 1:
                return "Recent (yesterday)"
            elif age.days <= 7:
                return f"Moderately recent ({age.days} days ago)"
            elif age.days <= 30:
                return f"Older ({age.days} days ago)"
            else:
                return f"Very old ({age.days} days ago)"
                
        except Exception as e:
            return f"Error calculating recency: {e}"
    
    def _explain_importance(self, memory_entry: Dict[str, Any]) -> str:
        """Explain importance score."""
        factors = []
        
        if memory_entry.get('importance_score'):
            factors.append("Has explicit importance score")
        
        entry_type = memory_entry.get('type', '')
        if entry_type:
            factors.append(f"Entry type: {entry_type}")
        
        if memory_entry.get('is_critical'):
            factors.append("Marked as critical")
        
        interaction_count = memory_entry.get('interaction_count', 0)
        if interaction_count > 0:
            factors.append(f"Has {interaction_count} interactions")
        
        user_rating = memory_entry.get('user_rating', 0)
        if user_rating > 0:
            factors.append(f"User rating: {user_rating}/5")
        
        return "; ".join(factors) if factors else "Default importance"
    
    def _explain_similarity(self, memory_content: str, current_task: str) -> str:
        """Explain similarity score."""
        if not memory_content or not current_task:
            return "No content to compare"
        
        if self.embeddings_available:
            return "Calculated using semantic embeddings"
        else:
            return "Calculated using keyword matching and phrase analysis"
    
    def _explain_role_relevance(self, memory_entry: Dict[str, Any], active_agent: str) -> str:
        """Explain role relevance score."""
        entry_roles = memory_entry.get('relevant_roles', [])
        if not entry_roles:
            entry_roles = self._infer_roles_from_content(memory_entry)
        
        if active_agent in entry_roles:
            return f"Direct match with active agent ({active_agent})"
        elif any(self._roles_are_related(active_agent, role) for role in entry_roles):
            return f"Related to active agent ({active_agent})"
        elif entry_roles:
            return f"Different roles: {', '.join(entry_roles)}"
        else:
            return "Role-agnostic content"
    
    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get statistics about scoring performance."""
        return {
            "embeddings_available": self.embeddings_available,
            "model_loaded": self.embedding_model is not None,
            "weights": self.weights.copy(),
            "version": "3.0"
        }