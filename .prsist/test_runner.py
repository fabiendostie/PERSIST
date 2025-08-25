#!/usr/bin/env python3
"""
Automated Test Runner for Prsist Memory System Phase 2-3
Runs comprehensive tests to verify all features are working correctly.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from memory_manager import MemoryManager
    from database import MemoryDatabase
    from enhanced_git_integration import EnhancedGitIntegrator, GitHookManager
    from productivity_tracker import ProductivityTracker
    from semantic_analyzer import SemanticAnalyzer
    from cross_session_correlator import CrossSessionCorrelator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the .prsist directory")
    sys.exit(1)


class TestResult:
    """Represents the result of a test."""
    def __init__(self, name: str, success: bool, message: str = "", duration: float = 0.0, data: Any = None):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration
        self.data = data
    
    def __str__(self):
        status = "✅" if self.success else "❌"
        return f"{status} {self.name} ({self.duration:.2f}s) - {self.message}"


class PrsistTestRunner:
    """Comprehensive test runner for Prsist Memory System."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.memory_dir = Path(__file__).parent
        self.repo_path = self.memory_dir.parent
        
    def log(self, message: str):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and record the result."""
        self.log(f"Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, dict) and result.get('success') is False:
                test_result = TestResult(test_name, False, result.get('message', 'Test failed'), duration, result)
            else:
                test_result = TestResult(test_name, True, "Test passed", duration, result)
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            test_result = TestResult(test_name, False, error_msg, duration)
            self.log(f"   Exception in {test_name}: {e}")
        
        self.results.append(test_result)
        self.log(f"   {test_result}")
        return test_result
    
    def test_basic_initialization(self) -> Dict[str, Any]:
        """Test basic system initialization."""
        try:
            # Test memory manager initialization
            manager = MemoryManager(str(self.memory_dir))
            
            # Test database connectivity
            db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
            
            # Test git integration (if available)
            try:
                git_integrator = EnhancedGitIntegrator(str(self.memory_dir), str(self.repo_path))
                git_available = True
            except:
                git_available = False
            
            return {
                'success': True,
                'memory_manager': True,
                'database': True,
                'git_integration': git_available
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_database_operations(self) -> Dict[str, Any]:
        """Test database operations."""
        try:
            db = MemoryDatabase(self.memory_dir / "storage" / "sessions.db")
            
            # Test session creation
            session_id = f"test_{int(time.time())}"
            success = db.create_session(session_id, str(self.repo_path), {'test': 'database_ops'})
            
            if not success:
                return {'success': False, 'message': 'Failed to create test session'}
            
            # Test session retrieval
            session = db.get_session(session_id)
            if not session:
                return {'success': False, 'message': 'Failed to retrieve test session'}
            
            # Test tool usage logging
            success = db.log_tool_usage(session_id, 'TestTool', {'test': 'input'}, {'test': 'output'})
            if not success:
                return {'success': False, 'message': 'Failed to log tool usage'}
            
            return {'success': True, 'session_id': session_id}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_git_integration(self) -> Dict[str, Any]:
        """Test git integration features."""
        try:
            integrator = EnhancedGitIntegrator(str(self.memory_dir), str(self.repo_path))
            
            # Test git state detection
            current_branch = integrator.current_branch
            current_commit = integrator.current_commit
            
            if not current_branch or not current_commit:
                return {'success': False, 'message': 'Could not detect git state'}
            
            # Test correlation updates check
            updates = integrator.check_for_correlation_updates()
            
            return {
                'success': True,
                'current_branch': current_branch,
                'current_commit': current_commit[:8] if current_commit else None,
                'updates_check': 'changes_detected' in updates
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Git integration failed: {str(e)}'}
    
    def test_session_correlation(self) -> Dict[str, Any]:
        """Test session correlation with git."""
        try:
            manager = MemoryManager(str(self.memory_dir))
            
            # Start a test session
            session = manager.start_session({'test': 'correlation'})
            session_id = session.get('session_id')
            
            if not session_id:
                return {'success': False, 'message': 'Failed to start test session'}
            
            # Check if git correlation occurred
            git_correlation = session.get('git_correlation', {})
            correlated = git_correlation.get('correlated', False)
            
            # End the session
            manager.end_session()
            
            return {
                'success': True,
                'session_id': session_id,
                'git_correlated': correlated,
                'correlation_data': git_correlation
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_semantic_analysis(self) -> Dict[str, Any]:
        """Test semantic analysis features."""
        try:
            analyzer = SemanticAnalyzer(str(self.memory_dir), str(self.repo_path))
            
            # Test file analysis on memory_manager.py
            test_file = self.memory_dir / "memory_manager.py"
            if not test_file.exists():
                return {'success': False, 'message': 'Test file memory_manager.py not found'}
            
            analysis = analyzer.analyze_file_semantics(str(test_file))
            
            if 'error' in analysis:
                return {'success': False, 'message': f'File analysis failed: {analysis["error"]}'}
            
            # Verify analysis contains expected elements
            expected_keys = ['language', 'elements', 'semantic_keywords', 'file_hash']
            missing_keys = [key for key in expected_keys if key not in analysis]
            
            if missing_keys:
                return {'success': False, 'message': f'Missing analysis keys: {missing_keys}'}
            
            return {
                'success': True,
                'language': analysis.get('language'),
                'element_count': len(analysis.get('elements', [])),
                'keyword_count': len(analysis.get('semantic_keywords', [])),
                'file_complexity': analysis.get('semantic_complexity', 0)
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_productivity_tracking(self) -> Dict[str, Any]:
        """Test productivity tracking features."""
        try:
            tracker = ProductivityTracker(str(self.memory_dir), str(self.repo_path))
            
            # Test velocity measurement
            velocity = tracker.measure_development_velocity(1)  # 1 day
            
            if 'error' in velocity:
                return {'success': False, 'message': f'Velocity measurement failed: {velocity["error"]}'}
            
            # Verify velocity contains expected data
            expected_keys = ['session_metrics', 'measurement_date', 'time_period_days']
            missing_keys = [key for key in expected_keys if key not in velocity]
            
            if missing_keys:
                return {'success': False, 'message': f'Missing velocity keys: {missing_keys}'}
            
            return {
                'success': True,
                'measurement_date': velocity.get('measurement_date'),
                'session_count': velocity.get('session_metrics', {}).get('total_sessions', 0),
                'has_git_metrics': 'git_metrics' in velocity
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_cross_session_correlation(self) -> Dict[str, Any]:
        """Test cross-session correlation features."""
        try:
            correlator = CrossSessionCorrelator(str(self.memory_dir), str(self.repo_path))
            
            # Test correlation building
            correlations = correlator.build_session_correlations(lookback_days=7)
            
            if 'error' in correlations:
                return {'success': False, 'message': f'Correlation failed: {correlations["error"]}'}
            
            # Verify correlation contains expected data
            expected_keys = ['sessions_analyzed', 'correlations_found', 'correlation_summary']
            missing_keys = [key for key in expected_keys if key not in correlations]
            
            if missing_keys:
                return {'success': False, 'message': f'Missing correlation keys: {missing_keys}'}
            
            return {
                'success': True,
                'sessions_analyzed': correlations.get('sessions_analyzed', 0),
                'correlations_found': correlations.get('correlations_found', 0),
                'clusters_found': len(correlations.get('correlation_clusters', []))
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_cli_bridge(self) -> Dict[str, Any]:
        """Test CLI bridge functionality."""
        try:
            manager = MemoryManager(str(self.memory_dir))
            
            # Test health check
            health = manager.cli_bridge_handler('health_check')
            if health.get('status') != 'success':
                return {'success': False, 'message': 'Health check failed'}
            
            # Test stats
            stats = manager.cli_bridge_handler('get_stats')
            if stats.get('status') != 'success':
                return {'success': False, 'message': 'Stats check failed'}
            
            # Test validation
            validation = manager.cli_bridge_handler('validate')
            if validation.get('status') != 'success':
                return {'success': False, 'message': 'Validation failed'}
            
            return {
                'success': True,
                'health_check': health.get('status'),
                'stats_available': bool(stats.get('data')),
                'validation_passed': validation.get('data', {}).get('valid', False)
            }
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling in various components."""
        try:
            manager = MemoryManager(str(self.memory_dir))
            
            # Test invalid session ID
            result = manager.cli_bridge_handler('get_context', ['invalid_session_123'])
            if result.get('status') != 'error':
                return {'success': False, 'message': 'Invalid session should return error'}
            
            # Test invalid command
            result = manager.cli_bridge_handler('invalid_command')
            if result.get('status') != 'error':
                return {'success': False, 'message': 'Invalid command should return error'}
            
            # Test semantic analyzer with invalid file
            analyzer = SemanticAnalyzer(str(self.memory_dir))
            result = analyzer.analyze_file_semantics('nonexistent_file.py')
            if 'error' not in result:
                return {'success': False, 'message': 'Invalid file should return error'}
            
            return {'success': True, 'error_handling': 'proper'}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance of key operations."""
        try:
            results = {}
            
            # Test memory manager initialization time
            start_time = time.time()
            manager = MemoryManager(str(self.memory_dir))
            results['init_time'] = time.time() - start_time
            
            # Test session start/end time
            start_time = time.time()
            session = manager.start_session({'test': 'performance'})
            session_id = session.get('session_id')
            manager.end_session()
            results['session_time'] = time.time() - start_time
            
            # Test semantic analysis time
            analyzer = SemanticAnalyzer(str(self.memory_dir))
            test_file = self.memory_dir / "memory_manager.py"
            
            if test_file.exists():
                start_time = time.time()
                analyzer.analyze_file_semantics(str(test_file))
                results['analysis_time'] = time.time() - start_time
            
            # Performance thresholds (in seconds)
            thresholds = {
                'init_time': 5.0,
                'session_time': 10.0,
                'analysis_time': 15.0
            }
            
            # Check if any operation is too slow
            slow_operations = []
            for operation, duration in results.items():
                if duration > thresholds.get(operation, 30.0):
                    slow_operations.append(f"{operation}: {duration:.2f}s")
            
            if slow_operations:
                return {
                    'success': False, 
                    'message': f'Slow operations detected: {", ".join(slow_operations)}',
                    'results': results
                }
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        self.log("🚀 Starting Prsist Memory System Test Suite")
        self.log("=" * 50)
        
        # Define test suite
        test_suite = [
            ("System Initialization", self.test_basic_initialization),
            ("Database Operations", self.test_database_operations),
            ("Git Integration", self.test_git_integration),
            ("Session Correlation", self.test_session_correlation),
            ("Semantic Analysis", self.test_semantic_analysis),
            ("Productivity Tracking", self.test_productivity_tracking),
            ("Cross-Session Correlation", self.test_cross_session_correlation),
            ("CLI Bridge", self.test_cli_bridge),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        # Run all tests
        for test_name, test_func in test_suite:
            self.run_test(test_name, test_func)
        
        # Calculate results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate summary
        self.log("=" * 50)
        self.log("📊 Test Results Summary")
        self.log(f"   Total Tests: {total_tests}")
        self.log(f"   Passed: {passed_tests} ✅")
        self.log(f"   Failed: {failed_tests} ❌")
        self.log(f"   Success Rate: {success_rate:.1f}%")
        
        # Show failed tests
        if failed_tests > 0:
            self.log("\n❌ Failed Tests:")
            for result in self.results:
                if not result.success:
                    self.log(f"   • {result.name}: {result.message}")
        
        # Overall status
        overall_success = success_rate >= 80  # 80% success rate threshold
        status = "✅ PASS" if overall_success else "❌ FAIL"
        self.log(f"\n🎯 Overall Status: {status}")
        
        return {
            'overall_success': overall_success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'results': [
                {
                    'name': r.name,
                    'success': r.success,
                    'message': r.message,
                    'duration': r.duration,
                    'data': r.data
                }
                for r in self.results
            ]
        }


def main():
    """Main entry point for test runner."""
    print("🧪 Prsist Memory System - Automated Test Runner")
    print("Phase 2-3 Feature Verification")
    print()
    
    # Check if we're in the right directory
    if not Path('.prsist').exists():
        print("❌ Error: Please run this from your project root directory")
        print("   The .prsist directory should be visible from here")
        sys.exit(1)
    
    # Initialize and run tests
    runner = PrsistTestRunner()
    
    try:
        results = runner.run_all_tests()
        
        # Save results to file
        results_file = Path('.prsist/docs/test_results.json')
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        runner.log(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()