#!/usr/bin/env python3
"""
Claude Code Integration Script for Prsist Memory System
Transparent integration that auto-launches and provides context
"""

import sys
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Configure minimal logging to avoid noise
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaudeCodeIntegration:
    def __init__(self):
        self.prsist_root = Path(__file__).parent.parent
        self.project_root = Path.cwd()
        self.config_file = self.prsist_root / 'config' / 'session-start.json'
        
        # Load configuration
        self.config = self.load_config()
        
        # Check if integration is enabled
        self.enabled = self.config.get('claude_code_integration', {}).get('enabled', True)
        self.transparent = self.config.get('claude_code_integration', {}).get('transparent_mode', True)
        
    def load_config(self):
        """Load session start configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
        return {}
    
    def is_prsist_available(self):
        """Check if prsist system is available"""
        try:
            prsist_script = self.prsist_root / 'bin' / 'prsist.py'
            return prsist_script.exists()
        except:
            return False
    
    def start_memory_session(self):
        """Start a new memory session silently"""
        try:
            if not self.is_prsist_available():
                return {"success": False, "reason": "prsist_not_available"}
            
            # Start session with Claude Code metadata
            session_metadata = {
                "tool": "claude-code",
                "session_type": "interactive_coding", 
                "project_root": str(self.project_root),
                "start_time": datetime.now().isoformat(),
                "auto_started": True
            }
            
            # Execute session start
            prsist_script = self.prsist_root / 'bin' / 'prsist.py'
            result = subprocess.run([
                sys.executable, str(prsist_script), '-n'
            ], 
            input=json.dumps(session_metadata),
            text=True, capture_output=True, timeout=10)
            
            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr, "returncode": result.returncode}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Session start timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_context(self):
        """Generate context for Claude injection with improved robustness"""
        try:
            if not self.is_prsist_available():
                return {"success": False, "reason": "prsist_not_available"}
            
            # Run context injector with longer timeout for startup
            injector_script = self.prsist_root / 'hooks' / 'claude-context-injector.py'
            result = subprocess.run([
                sys.executable, str(injector_script)
            ], capture_output=True, text=True, timeout=8)
            
            if result.returncode == 0:
                try:
                    context_result = json.loads(result.stdout)
                    return {"success": True, "context": context_result}
                except json.JSONDecodeError:
                    return {"success": True, "context": {"raw_output": result.stdout}}
            else:
                # Even if context generation fails, we still succeeded in starting
                return {"success": True, "context": {"error": result.stderr, "fallback": True}}
                
        except subprocess.TimeoutExpired:
            return {"success": True, "context": {"error": "Context generation timeout", "fallback": True}}
        except Exception as e:
            return {"success": True, "context": {"error": str(e), "fallback": True}}
    
    def run_integration(self):
        """Run full Claude Code integration"""
        results = {
            "integration_status": "starting",
            "prsist_available": False,
            "session_started": False,
            "context_generated": False,
            "timestamp": datetime.now().isoformat(),
            "messages": []
        }
        
        try:
            # Check if integration is enabled
            if not self.enabled:
                results["integration_status"] = "disabled"
                results["messages"].append("Prsist integration is disabled in configuration")
                return results
            
            # Check system availability
            results["prsist_available"] = self.is_prsist_available()
            if not results["prsist_available"]:
                results["integration_status"] = "unavailable"
                results["messages"].append("Prsist system not found - running without memory")
                return results
            
            # Start memory session
            session_result = self.start_memory_session()
            results["session_started"] = session_result["success"]
            if session_result["success"]:
                results["messages"].append("Memory session started successfully")
            else:
                results["messages"].append(f"Session start failed: {session_result.get('error', 'unknown error')}")
            
            # Generate context
            context_result = self.generate_context()
            results["context_generated"] = context_result["success"]
            if context_result["success"]:
                results["context_file"] = context_result["context"].get("context_file")
                results["messages"].append("Context generated for Claude Code")
            else:
                results["messages"].append(f"Context generation failed: {context_result.get('error', 'unknown error')}")
            
            # Set final status
            if results["session_started"] or results["context_generated"]:
                results["integration_status"] = "active"
                results["messages"].append("Prsist memory system is ready")
            else:
                results["integration_status"] = "failed"
                results["messages"].append("Integration failed - Claude will work without memory")
            
            return results
            
        except Exception as e:
            results["integration_status"] = "error"
            results["messages"].append(f"Integration error: {str(e)}")
            return results

def main():
    """Main entry point for Claude Code integration"""
    integration = ClaudeCodeIntegration()
    
    # Run integration
    results = integration.run_integration()
    
    # Always provide clear status output (ASCII-safe for Windows)
    status_message = ""
    if results["integration_status"] == "active":
        status_message = "[SUCCESS] Prsist Memory System ready"
    elif results["integration_status"] == "disabled":
        status_message = "[DISABLED] Prsist Memory System disabled"
    elif results["integration_status"] == "unavailable":
        status_message = "[WARNING] Prsist Memory System unavailable (Claude will work normally)"
    elif results["integration_status"] == "failed":
        error_msg = next((msg for msg in results["messages"] if "failed" in msg or "error" in msg.lower()), "Unknown error")
        status_message = f"[ERROR] Prsist Memory System error: {error_msg}"
    else:
        status_message = f"[STATUS] Prsist Memory System status: {results['integration_status']}"
    
    # Output to console
    print(status_message)
    
    # Also write to log file for debugging (since Claude Code may not show SessionStart output)
    try:
        log_file = integration.prsist_root / "logs" / "integration.log"
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}: {status_message}\n")
    except:
        pass  # Silent failure for logging
    
    # Show additional context in verbose mode only
    if not integration.transparent and results["integration_status"] not in ["active", "disabled"]:
        print("\nDetailed results:")
        print(json.dumps(results, indent=2))
    
    # Always return success to not block Claude Code
    return 0

if __name__ == "__main__":
    sys.exit(main())