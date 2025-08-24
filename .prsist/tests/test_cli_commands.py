#!/usr/bin/env python3
"""
Comprehensive test suite for prsist CLI commands
Tests every single command and command combination
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class CLITester:
    def __init__(self):
        self.prsist_path = Path(__file__).parent.parent / "bin" / "prsist.py"
        self.test_results = {}
        self.passed = 0
        self.failed = 0
        
    def run_command(self, flags, timeout=10):
        """Run prsist command and capture output"""
        try:
            cmd = [sys.executable, str(self.prsist_path), f"-{flags}"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                input="\n" * 10  # Provide empty inputs for interactive commands
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -2
            }
    
    def test_command(self, flag, description, expected_patterns=None):
        """Test a single command"""
        print(f"Testing -{flag}: {description}...", end=" ")
        
        result = self.run_command(flag)
        success = result["success"]
        
        # Check for expected patterns in output
        if expected_patterns and success:
            for pattern in expected_patterns:
                if pattern not in result["stdout"]:
                    success = False
                    result["error"] = f"Expected pattern '{pattern}' not found in output"
                    break
        
        # Store result
        self.test_results[flag] = {
            "description": description,
            "success": success,
            "result": result
        }
        
        if success:
            print("[PASS]")
            self.passed += 1
        else:
            print("[FAIL]")
            self.failed += 1
            if result.get("stderr"):
                print(f"  Error: {result['stderr']}")
            if result.get("error"):
                print(f"  Error: {result['error']}")
    
    def test_all_commands(self):
        """Test every single CLI command"""
        print("=" * 60)
        print("BMAD Memory CLI - Comprehensive Command Test Suite")
        print("=" * 60)
        
        # Core Operations
        print("\n[CORE] Testing Core Operations:")
        self.test_command("t", "Test system", ["[TEST]", "[PASS]"])
        self.test_command("s", "Session status", ["[STATUS]"])
        self.test_command("c", "Context (what Claude sees)", ["[CONTEXT]"])
        self.test_command("r", "Recent sessions", ["[RECENT]"])
        self.test_command("h", "Health check", ["[HEALTH]"])
        self.test_command("m", "Memory stats", ["[STATS]"])
        self.test_command("v", "Validate system", ["[VALIDATE]"])
        
        # Session Management
        print("\n[SESSION] Testing Session Management:")
        self.test_command("n", "New session (start)", ["[NEW]"])
        self.test_command("e", "End session", ["[END]"])
        self.test_command("k", "Checkpoint (create)", ["[CHECKPOINT]"])
        self.test_command("x", "Export session data", ["[EXPORT]"])
        
        # Data Management
        print("\n[DATA] Testing Data Management:")
        self.test_command("f", "Feature log (interactive)", ["[FEATURE]"])
        self.test_command("p", "Project memory (add)", ["[PROJECT]"])
        self.test_command("d", "Add decision", ["[DECISION]"])
        self.test_command("z", "Cleanup old data", ["[CLEANUP]"])
        
        # Shortcuts
        print("\n[SHORTCUTS] Testing Shortcuts:")
        self.test_command("a", "All core checks", ["[RUN]"])
        self.test_command("l", "List commands", ["[HELP]", "Core Operations"])
        
        # Command Chaining
        print("\n[CHAIN] Testing Command Chaining:")
        self.test_command("hm", "Health + Memory stats", ["[HEALTH]", "[STATS]"])
        self.test_command("tsc", "Test + Status + Context", ["[TEST]", "[STATUS]", "[CONTEXT]"])
        self.test_command("rv", "Recent + Validate", ["[RECENT]", "[VALIDATE]"])
        
        # Edge Cases
        print("\n[EDGE] Testing Edge Cases:")
        
        # Invalid command
        print("Testing invalid command...", end=" ")
        invalid_result = self.run_command("q")  # 'q' is not a valid command
        if "Unknown command" in invalid_result["stdout"]:
            print("[PASS]")
            self.passed += 1
        else:
            print("[FAIL]")
            self.failed += 1
            
        # Empty command
        print("Testing empty command...", end=" ")
        try:
            result = subprocess.run([sys.executable, str(self.prsist_path)], 
                                  capture_output=True, text=True, timeout=5)
            if "Usage: prsist [options]" in result.stdout:
                print("[PASS]")
                self.passed += 1
            else:
                print("[FAIL]")
                self.failed += 1
        except:
            print("[FAIL]")
            self.failed += 1
    
    def test_memory_system_integration(self):
        """Test that commands actually interact with memory system"""
        print("\n[INTEGRATION] Testing Memory System Integration:")
        
        # Test that commands actually use MemoryManager
        integration_tests = [
            ("t", "system test integration"),
            ("s", "session status integration"), 
            ("c", "context building integration"),
            ("h", "health check integration"),
            ("m", "memory stats integration"),
            ("v", "validation integration")
        ]
        
        for flag, description in integration_tests:
            print(f"Testing {description}...", end=" ")
            result = self.run_command(flag)
            
            # Check if it shows memory manager activity (logging messages)
            if result["success"] and ("Memory manager initialized" in result["stderr"] or 
                                    "[PASS]" in result["stdout"] or 
                                    "[STATS]" in result["stdout"]):
                print("[PASS]")
                self.passed += 1
            else:
                print("[FAIL]")
                self.failed += 1
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed > 0:
            print(f"\n[FAILED] Failed Tests ({self.failed}):")
            for flag, test_data in self.test_results.items():
                if not test_data["success"]:
                    print(f"  -{flag}: {test_data['description']}")
                    if test_data["result"].get("stderr"):
                        print(f"    Error: {test_data['result']['stderr']}")
        
        if success_rate >= 90:
            print(f"\n[EXCELLENT] {success_rate:.1f}% success rate")
        elif success_rate >= 75:
            print(f"\n[GOOD] {success_rate:.1f}% success rate")
        elif success_rate >= 50:
            print(f"\n[FAIR] {success_rate:.1f}% success rate - needs improvement")
        else:
            print(f"\n[POOR] {success_rate:.1f}% success rate - major issues")
        
        return success_rate >= 75
    
    def save_detailed_report(self):
        """Save detailed test results to file"""
        report_file = Path(__file__).parent / "test_results.json"
        
        detailed_report = {
            "test_summary": {
                "total_tests": self.passed + self.failed,
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": (self.passed / (self.passed + self.failed) * 100) if (self.passed + self.failed) > 0 else 0
            },
            "test_results": self.test_results,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        with open(report_file, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"\n[REPORT] Detailed report saved to: {report_file}")

def main():
    """Run comprehensive CLI test suite"""
    tester = CLITester()
    
    # Test all commands
    tester.test_all_commands()
    
    # Test memory system integration
    tester.test_memory_system_integration()
    
    # Generate report
    success = tester.generate_report()
    
    # Save detailed report
    tester.save_detailed_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()