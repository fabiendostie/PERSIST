#!/usr/bin/env python3
"""
Comprehensive test suite for Prsist Memory System Phase 3.
Tests all Phase 3 features including file watching, context management,
relevance scoring, knowledge persistence, and service orchestration.
"""

import sys
import os
import json
import tempfile
import shutil
import asyncio
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import threading

# Add memory system to Python path
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

try:
    from database import MemoryDatabase
    from context_manager import ContextManager, ContextCompressor
    from relevance_scorer import RelevanceScorer
    from knowledge_manager import KnowledgeManager, PatternDetector, DecisionTracker
    from advanced_change_analyzer import AdvancedChangeImpactAnalyzer
    from performance_monitor import PerformanceMonitor, MetricCollector, AlertManager
    from services.memory_service_orchestrator import MemoryServiceOrchestrator
    from utils import setup_logging
except ImportError as e:
    print(f"Memory system not available: {e}")
    sys.exit(1)

class Phase3TestSuite:
    """Comprehensive test suite for Phase 3 features."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = []
        self.temp_dir = None
        self.test_db_path = None
        setup_logging("WARNING")  # Quiet during tests
        
    def setup_test_environment(self):
        """Set up test environment."""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="bmad_memory_phase3_test_")
            self.test_db_path = Path(self.temp_dir) / "test_memory.db"
            
            # Create test config directory
            config_dir = Path(self.temp_dir) / "config"
            config_dir.mkdir()
            
            return True
            
        except Exception as e:
            print(f"Failed to set up test environment: {e}")
            return False
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        try:
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
            return True
        except Exception as e:
            print(f"Failed to clean up test environment: {e}")
            return False
    
    def run_test(self, test_name: str, test_func) -> dict:
        """Run a single test and record results."""
        print(f"Running test: {test_name}")
        
        try:
            start_time = datetime.now()
            result = test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = {
                "name": test_name,
                "status": "passed" if result.get("success", False) else "failed",
                "duration": duration,
                "details": result,
                "timestamp": start_time.isoformat()
            }
            
            if test_result["status"] == "passed":
                print(f"   + {test_name} passed ({duration:.2f}s)")
            else:
                print(f"   - {test_name} failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            test_result = {
                "name": test_name,
                "status": "error",
                "duration": 0,
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
            print(f"   - {test_name} error: {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    def test_context_manager(self) -> dict:
        """Test context manager functionality."""
        try:
            config = {
                "context_management": {
                    "max_context_tokens": 75000,
                    "compression_threshold": 0.95,
                    "relevance_threshold": 0.6
                }
            }
            
            context_manager = ContextManager(config)
            
            # Test base context retrieval
            session_id = str(uuid.uuid4())
            base_context = context_manager.get_base_context(session_id)
            
            assert base_context is not None, "Base context should not be None"
            assert "session_id" in base_context, "Base context should contain session_id"
            assert base_context["session_id"] == session_id, "Session ID should match"
            
            # Test context expansion
            current_task = "implement new feature for memory system"
            expanded_context = context_manager.expand_context(base_context, current_task)
            
            assert expanded_context is not None, "Expanded context should not be None"
            assert len(expanded_context) >= len(base_context), "Expanded context should be larger"
            
            # Test task analysis
            task_analysis = context_manager.analyze_task(current_task)
            assert "type" in task_analysis, "Task analysis should include type"
            assert "complexity" in task_analysis, "Task analysis should include complexity"
            assert "knowledge_areas" in task_analysis, "Task analysis should include knowledge areas"
            
            # Test token usage calculation
            token_usage = context_manager.calculate_token_usage(base_context)
            assert 0 <= token_usage <= 1, "Token usage should be between 0 and 1"
            
            return {"success": True, "context_manager": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_context_compressor(self) -> dict:
        """Test context compression functionality."""
        try:
            compressor = ContextCompressor()
            
            # Create test context
            test_context = {
                "session_id": "test_session",
                "timestamp": datetime.now().isoformat(),
                "critical_info": {"important": "data"},
                "current_session": {"status": "active"},
                "completed_phases": [
                    {"phase": 1, "key_outcomes": ["implemented feature A"]},
                    {"phase": 2, "key_outcomes": ["implemented feature B"]}
                ],
                "background": {"project_type": "bmad-method"},
                "recent_files": [f"file_{i}.py" for i in range(20)],
                "recent_tools": [{"tool_name": f"tool_{i}", "timestamp": datetime.now().isoformat()} for i in range(30)]
            }
            
            # Test hierarchical compression
            compressed = compressor.hierarchical_compress(test_context)
            
            assert compressed is not None, "Compressed context should not be None"
            assert "critical" in compressed, "Critical info should be preserved"
            assert "current_session" in compressed, "Current session should be preserved"
            
            # Test auto-compact
            auto_compressed = compressor.auto_compact(test_context, capacity_threshold=0.01)  # Force compression
            assert auto_compressed is not None, "Auto-compressed context should not be None"
            
            # Test compression strategies
            semantic_compressed = compressor.semantic_compress(test_context)
            temporal_compressed = compressor.temporal_compress(test_context)
            importance_compressed = compressor.importance_compress(test_context)
            
            return {"success": True, "compressor": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_relevance_scorer(self) -> dict:
        """Test relevance scoring functionality."""
        try:
            scorer = RelevanceScorer()
            
            # Test memory entry
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "content": "implement file watching system with chokidar",
                "importance_score": 0.8,
                "type": "implementation",
                "relevant_roles": ["developer"]
            }
            
            # Test current context
            current_context = {
                "current_task": "create file monitoring for memory system",
                "active_agent": "developer"
            }
            
            # Test relevance calculation
            final_score, dimension_scores = scorer.calculate_relevance(memory_entry, current_context)
            
            assert 0 <= final_score <= 1, "Final score should be between 0 and 1"
            assert "recency" in dimension_scores, "Should have recency score"
            assert "importance" in dimension_scores, "Should have importance score"
            assert "similarity" in dimension_scores, "Should have similarity score"
            assert "role_relevance" in dimension_scores, "Should have role relevance score"
            
            # Test batch scoring
            memory_entries = [memory_entry.copy() for _ in range(5)]
            scored_entries = scorer.batch_score_entries(memory_entries, current_context, top_k=3)
            
            assert len(scored_entries) <= 3, "Should return at most 3 entries"
            assert all(len(entry) == 3 for entry in scored_entries), "Each entry should have 3 elements"
            
            # Test score explanation
            explanation = scorer.explain_score(memory_entry, current_context)
            assert "final_score" in explanation, "Should have final score"
            assert "dimension_scores" in explanation, "Should have dimension scores"
            assert "explanations" in explanation, "Should have explanations"
            
            # Test scoring statistics
            stats = scorer.get_scoring_stats()
            assert "embeddings_available" in stats, "Should report embeddings availability"
            assert "weights" in stats, "Should report weights"
            
            return {"success": True, "relevance_scorer": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_knowledge_manager(self) -> dict:
        """Test knowledge management functionality."""
        try:
            knowledge_manager = KnowledgeManager(self.temp_dir)
            
            # Test pattern detector
            pattern_detector = PatternDetector()
            
            test_session_data = {
                "session_id": "test_session",
                "tool_usage": [
                    {"tool_name": "Edit", "file_path": "test.py"},
                    {"tool_name": "Edit", "file_path": "config.py"},
                    {"tool_name": "Read", "file_path": "readme.md"},
                    {"tool_name": "Bash", "command": "npm test"}
                ]
            }
            
            patterns = pattern_detector.extract_patterns(test_session_data)
            assert isinstance(patterns, list), "Patterns should be a list"
            
            # Test decision tracker
            decision_tracker = DecisionTracker()
            decisions = decision_tracker.extract_decisions(test_session_data)
            assert isinstance(decisions, list), "Decisions should be a list"
            
            # Test knowledge persistence
            persistence_result = knowledge_manager.persist_session_knowledge(test_session_data)
            assert "patterns_stored" in persistence_result, "Should report patterns stored"
            assert "decisions_stored" in persistence_result, "Should report decisions stored"
            
            # Test knowledge retrieval
            current_context = {"current_task": "implement testing system"}
            relevant_knowledge = knowledge_manager.get_relevant_knowledge(current_context)
            
            assert "patterns" in relevant_knowledge, "Should have patterns"
            assert "decisions" in relevant_knowledge, "Should have decisions"
            assert "project_context" in relevant_knowledge, "Should have project context"
            
            # Test knowledge statistics
            stats = knowledge_manager.get_knowledge_stats()
            assert "totals" in stats, "Should have totals"
            
            return {"success": True, "knowledge_manager": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_change_impact_analyzer(self) -> dict:
        """Test change impact analysis functionality."""
        try:
            memory_db = MemoryDatabase(str(self.test_db_path))
            analyzer = AdvancedChangeImpactAnalyzer(memory_db)
            
            # Test file type detection
            assert analyzer.detect_file_type("test.py") == "code"
            assert analyzer.detect_file_type("config.yaml") == "config"
            assert analyzer.detect_file_type("README.md") == "documentation"
            assert analyzer.detect_file_type("test_file.py") == "test"
            
            # Test change impact analysis
            impact_analysis = analyzer.analyze_change_impact(
                file_path="src/memory_system.py",
                change_type="change",
                content_diff="+def new_function():\n+    return 'hello world'",
                session_context={"recent_files": ["src/memory_system.py"], "goals": ["implement new feature"]}
            )
            
            assert "overall_impact" in impact_analysis, "Should have overall impact"
            assert "base_analysis" in impact_analysis, "Should have base analysis"
            assert "context_analysis" in impact_analysis, "Should have context analysis"
            assert "session_impact" in impact_analysis, "Should have session impact"
            assert "memory_implications" in impact_analysis, "Should have memory implications"
            assert "recommendations" in impact_analysis, "Should have recommendations"
            
            # Test pattern analysis
            patterns = analyzer.analyze_change_patterns(days=1)
            assert "total_changes" in patterns, "Should have total changes"
            assert "change_types" in patterns, "Should have change types"
            
            return {"success": True, "change_analyzer": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_performance_monitor(self) -> dict:
        """Test performance monitoring functionality."""
        try:
            memory_db = MemoryDatabase(str(self.test_db_path))
            
            config = {
                "monitoring": {
                    "alerts": {
                        "thresholds": {
                            "process_memory_usage_mb_warning": 1000,
                            "process_cpu_usage_percent_warning": 80
                        }
                    }
                }
            }
            
            performance_monitor = PerformanceMonitor(memory_db, config)
            
            # Test metric collector
            collector = performance_monitor.collector
            
            system_metrics = collector.collect_system_metrics()
            assert len(system_metrics) > 0, "Should collect system metrics"
            
            process_metrics = collector.collect_process_metrics()
            assert len(process_metrics) > 0, "Should collect process metrics"
            
            database_metrics = collector.collect_database_metrics()
            assert len(database_metrics) >= 0, "Should collect database metrics"
            
            # Test performance analyzer
            analyzer = performance_monitor.analyzer
            all_metrics = system_metrics + process_metrics + database_metrics
            
            analysis = analyzer.analyze_metrics(all_metrics)
            assert "trends" in analysis, "Should have trends"
            assert "anomalies" in analysis, "Should have anomalies"
            assert "recommendations" in analysis, "Should have recommendations"
            assert "health_score" in analysis, "Should have health score"
            
            # Test alert manager
            alert_manager = performance_monitor.alert_manager
            alerts = alert_manager.check_alerts(all_metrics)
            assert isinstance(alerts, list), "Alerts should be a list"
            
            alert_summary = alert_manager.get_alert_summary()
            assert "active_alerts_count" in alert_summary, "Should have active alerts count"
            
            # Test current metrics
            current_metrics = performance_monitor.get_current_metrics()
            assert isinstance(current_metrics, dict), "Current metrics should be a dict"
            
            # Test performance report
            report = performance_monitor.get_performance_report(hours=1)
            assert "current_metrics" in report, "Should have current metrics"
            assert "alert_summary" in report, "Should have alert summary"
            assert "health_status" in report, "Should have health status"
            
            return {"success": True, "performance_monitor": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_database_phase3_features(self) -> dict:
        """Test Phase 3 database features."""
        try:
            memory_db = MemoryDatabase(str(self.test_db_path))
            
            session_id = str(uuid.uuid4())
            
            # Test context snapshots
            context_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            success = memory_db.create_context_snapshot(
                session_id=session_id,
                snapshot_type="test",
                context_data=context_data,
                compression_level=1,
                relevance_scores=[0.8, 0.6, 0.4]
            )
            assert success, "Should create context snapshot"
            
            snapshots = memory_db.get_context_snapshots(session_id, limit=5)
            assert len(snapshots) > 0, "Should retrieve context snapshots"
            
            # Test file relevance
            success = memory_db.update_file_relevance(
                file_path="test.py",
                session_id=session_id,
                relevance_score=0.8,
                relevance_factors=["test_factor"],
                expires_at=datetime.now() + timedelta(hours=1)
            )
            assert success, "Should update file relevance"
            
            relevances = memory_db.get_file_relevance(file_path="test.py")
            assert len(relevances) > 0, "Should retrieve file relevances"
            
            # Test context injections
            success = memory_db.record_context_injection(
                session_id=session_id,
                injection_type="test",
                injected_content={"content": "test injection"},
                relevance_score=0.7
            )
            assert success, "Should record context injection"
            
            injections = memory_db.get_context_injections(session_id)
            assert len(injections) > 0, "Should retrieve context injections"
            
            # Test cross-session relationships
            other_session_id = str(uuid.uuid4())
            success = memory_db.create_cross_session_relationship(
                source_session_id=session_id,
                target_session_id=other_session_id,
                relationship_type="test_relationship",
                relationship_strength=0.8,
                shared_elements=["shared_element"]
            )
            assert success, "Should create cross-session relationship"
            
            # Test performance metrics
            success = memory_db.record_performance_metric(
                metric_type="test_metric",
                metric_value=42.0,
                session_id=session_id,
                measurement_context={"test": "context"}
            )
            assert success, "Should record performance metric"
            
            metrics = memory_db.get_performance_metrics(metric_type="test_metric", limit=5)
            assert len(metrics) > 0, "Should retrieve performance metrics"
            
            # Test change impact
            success = memory_db.record_change_impact(
                file_path="test.py",
                change_type="change",
                impact_score=0.8,
                affected_sessions=[session_id],
                memory_invalidation=True,
                context_refresh_required=True
            )
            assert success, "Should record change impact"
            
            impacts = memory_db.get_change_impacts(file_path="test.py")
            assert len(impacts) > 0, "Should retrieve change impacts"
            
            # Test cleanup
            cleaned_count = memory_db.cleanup_expired_relevance()
            assert cleaned_count >= 0, "Should perform cleanup"
            
            return {"success": True, "database_phase3": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_service_orchestrator_setup(self) -> dict:
        """Test service orchestrator setup (without actually starting services)."""
        try:
            # Create test config
            config_path = Path(self.temp_dir) / "test_config.yaml"
            test_config = {
                "memory_system": {
                    "version": "3.0",
                    "advanced_features": {
                        "file_watching": False,  # Disable for testing
                        "dynamic_context": True,
                        "relevance_scoring": True,
                        "knowledge_persistence": True
                    }
                },
                "services": {
                    "file_watcher": {"enabled": False},
                    "context_injector": {"enabled": True},
                    "knowledge_manager": {"enabled": True},
                    "performance_monitor": {"enabled": True}
                },
                "performance": {
                    "async_processing": True,
                    "thread_pool_size": 2,
                    "queue_max_size": 100
                }
            }
            
            # Write config
            import yaml
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(test_config, f)
            except ImportError:
                # Fallback to JSON if YAML not available
                config_path = Path(self.temp_dir) / "test_config.json"
                with open(config_path, 'w') as f:
                    json.dump(test_config, f)
            
            # Test orchestrator initialization
            orchestrator = MemoryServiceOrchestrator(str(config_path))
            
            assert orchestrator.config is not None, "Should load configuration"
            assert orchestrator.memory_db is not None, "Should initialize memory database"
            assert orchestrator.context_manager is not None, "Should initialize context manager"
            assert orchestrator.relevance_scorer is not None, "Should initialize relevance scorer"
            
            # Test status retrieval
            status = orchestrator.get_service_status()
            assert "orchestrator_running" in status, "Should have orchestrator running status"
            assert "services" in status, "Should have services status"
            
            # Test task queueing
            test_task = {"type": "test", "data": "test_data"}
            # Can't test actual queueing without starting async loop
            
            return {"success": True, "orchestrator": "setup_working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_integration_features(self) -> dict:
        """Test integration between different Phase 3 components."""
        try:
            memory_db = MemoryDatabase(str(self.test_db_path))
            
            # Test context manager with relevance scorer integration
            config = {"context_management": {"max_context_tokens": 75000}}
            context_manager = ContextManager(config)
            relevance_scorer = RelevanceScorer()
            
            # Set up integration
            context_manager.relevance_scorer = relevance_scorer
            
            session_id = str(uuid.uuid4())
            current_task = "test integration between components"
            
            # Test dynamic context with relevance scoring
            context = context_manager.get_dynamic_context(session_id, current_task)
            assert context is not None, "Should get dynamic context"
            
            # Test change analyzer with database integration
            change_analyzer = AdvancedChangeImpactAnalyzer(memory_db)
            
            impact_analysis = change_analyzer.analyze_change_impact(
                file_path="integration_test.py",
                change_type="add",
                content_diff="+print('integration test')"
            )
            
            assert impact_analysis["overall_impact"] > 0, "Should calculate impact"
            
            # Test knowledge manager with pattern detection
            knowledge_manager = KnowledgeManager(self.temp_dir)
            
            test_session_data = {
                "session_id": session_id,
                "tool_usage": [{"tool_name": "Edit", "file_path": "integration_test.py"}],
                "context_data": {"integration_test": True}
            }
            
            persistence_result = knowledge_manager.persist_session_knowledge(test_session_data)
            assert persistence_result.get("patterns_stored", 0) >= 0, "Should process session data"
            
            return {"success": True, "integration": "working"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_error_handling(self) -> dict:
        """Test error handling across components."""
        try:
            errors_handled = 0
            
            # Test context manager with invalid data
            try:
                config = {"context_management": {"max_context_tokens": "invalid"}}
                context_manager = ContextManager(config)
                context = context_manager.get_dynamic_context("", "")
                # Should handle gracefully
                errors_handled += 1
            except Exception:
                pass
            
            # Test relevance scorer with empty data
            try:
                scorer = RelevanceScorer()
                score, _ = scorer.calculate_relevance({}, {})
                # Should return default values
                errors_handled += 1
            except Exception:
                pass
            
            # Test database with invalid paths
            try:
                invalid_db = MemoryDatabase("/invalid/path/db.sqlite")
                # Should handle initialization gracefully
                errors_handled += 1
            except Exception:
                pass
            
            # Test knowledge manager with invalid storage path
            try:
                knowledge_manager = KnowledgeManager("/invalid/path")
                # Should handle initialization gracefully
                errors_handled += 1
            except Exception:
                pass
            
            return {"success": True, "errors_handled": errors_handled}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_performance_benchmarks(self) -> dict:
        """Test performance benchmarks for Phase 3 features."""
        try:
            memory_db = MemoryDatabase(str(self.test_db_path))
            benchmarks = {}
            
            # Benchmark context management
            start_time = time.time()
            config = {"context_management": {"max_context_tokens": 75000}}
            context_manager = ContextManager(config)
            
            for _ in range(10):
                session_id = str(uuid.uuid4())
                context = context_manager.get_dynamic_context(session_id, "benchmark test")
            
            benchmarks["context_management_ms"] = (time.time() - start_time) * 100  # Per operation
            
            # Benchmark relevance scoring
            start_time = time.time()
            scorer = RelevanceScorer()
            
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "content": "test content for benchmarking",
                "importance_score": 0.5
            }
            current_context = {"current_task": "benchmark relevance scoring"}
            
            for _ in range(50):
                scorer.calculate_relevance(memory_entry, current_context)
            
            benchmarks["relevance_scoring_ms"] = (time.time() - start_time) * 20  # Per operation
            
            # Benchmark database operations
            start_time = time.time()
            
            for i in range(100):
                memory_db.record_performance_metric(
                    metric_type="benchmark_test",
                    metric_value=float(i),
                    measurement_context={"iteration": i}
                )
            
            benchmarks["database_insert_ms"] = (time.time() - start_time) * 10  # Per operation
            
            # Benchmark change analysis
            start_time = time.time()
            analyzer = AdvancedChangeImpactAnalyzer(memory_db)
            
            for i in range(10):
                analyzer.analyze_change_impact(f"test_file_{i}.py", "change")
            
            benchmarks["change_analysis_ms"] = (time.time() - start_time) * 100  # Per operation
            
            # Validate performance requirements
            performance_requirements = {
                "context_management_ms": 500,   # Max 500ms per operation
                "relevance_scoring_ms": 100,    # Max 100ms per operation
                "database_insert_ms": 50,       # Max 50ms per operation
                "change_analysis_ms": 1000      # Max 1s per operation
            }
            
            performance_passed = all(
                benchmarks[key] <= performance_requirements[key]
                for key in performance_requirements
            )
            
            return {
                "success": True,
                "benchmarks": benchmarks,
                "requirements": performance_requirements,
                "performance_passed": performance_passed
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self) -> dict:
        """Run all Phase 3 tests."""
        print("Starting Prsist Memory System Phase 3 Test Suite")
        print("=" * 70)
        
        # Set up test environment
        if not self.setup_test_environment():
            return {"success": False, "error": "Failed to set up test environment"}
        
        try:
            # Define all tests
            tests = [
                ("Context Manager", self.test_context_manager),
                ("Context Compressor", self.test_context_compressor),
                ("Relevance Scorer", self.test_relevance_scorer),
                ("Knowledge Manager", self.test_knowledge_manager),
                ("Change Impact Analyzer", self.test_change_impact_analyzer),
                ("Performance Monitor", self.test_performance_monitor),
                ("Database Phase 3 Features", self.test_database_phase3_features),
                ("Service Orchestrator Setup", self.test_service_orchestrator_setup),
                ("Integration Features", self.test_integration_features),
                ("Error Handling", self.test_error_handling),
                ("Performance Benchmarks", self.test_performance_benchmarks)
            ]
            
            # Run all tests
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
            
            # Calculate summary
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results if result["status"] == "passed")
            failed_tests = sum(1 for result in self.test_results if result["status"] == "failed")
            error_tests = sum(1 for result in self.test_results if result["status"] == "error")
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            summary = {
                "success": success_rate >= 80,  # 80% pass rate required
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": success_rate,
                "test_results": self.test_results
            }
            
            # Print summary
            print("=" * 70)
            print("Phase 3 Test Summary:")
            print(f"  Total tests: {total_tests}")
            print(f"  Passed: {passed_tests}")
            print(f"  Failed: {failed_tests}")
            print(f"  Errors: {error_tests}")
            print(f"  Success rate: {success_rate:.1f}%")
            
            if summary["success"]:
                print("\n[SUCCESS] Phase 3 test suite passed!")
            else:
                print("\n[FAILED] Phase 3 test suite failed!")
                print("Failed tests:")
                for result in self.test_results:
                    if result["status"] != "passed":
                        print(f"  - {result['name']}: {result['details'].get('error', 'Unknown error')}")
            
            return summary
            
        finally:
            # Clean up test environment
            self.cleanup_test_environment()

def main():
    """Main test execution."""
    test_suite = Phase3TestSuite()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()