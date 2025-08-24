#!/usr/bin/env python3
"""
Claude Code Memory Commands
Usage: python claude-commands.py <command> [args...]

This enables *feature, *decision, *status style commands in Claude Code
"""

import sys
import os
import subprocess

# Add .prsist to path for direct imports
sys.path.insert(0, '.prsist')

def handle_command():
    if len(sys.argv) < 2:
        print("Memory System Commands (use with python claude-commands.py):")
        print("  status                           - Show session status")
        print("  health                           - Health check") 
        print("  memory                           - Memory stats")
        print("  context                          - Show context")
        print("  feature <name> <description>     - Log feature")
        print("  decision <text>                  - Add decision")
        print("  recent                           - Recent sessions")
        print("")
        print("For Claude Code integration, use:")
        print("  python claude-commands.py feature 'API Done' 'Completed REST API'")
        return 1
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    try:
        if command == 'feature':
            if len(args) < 2:
                print("Usage: python claude-commands.py feature 'Name' 'Description'")
                print("Example: python claude-commands.py feature 'Bug Fix' 'Fixed memory leak'")
                return 1
            
            name = args[0]
            description = ' '.join(args[1:])
            result = subprocess.run(['python', '.prsist/hooks/FeatureComplete.py', name, description], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Feature logged: {name}")
                print(f"Description: {description}")
                print("Checkpoint created automatically")
            else:
                print(f"Error logging feature: {result.stderr}")
            return result.returncode
        
        elif command == 'decision':
            if len(args) < 1:
                print("Usage: python claude-commands.py decision 'Your decision text'")
                print("Example: python claude-commands.py decision 'Use PostgreSQL for better performance'")
                return 1
            
            decision_text = ' '.join(args)
            
            # Import and use memory manager
            try:
                from memory_manager import MemoryManager
                mm = MemoryManager()
                result = mm.add_project_memory(f'**Decision**: {decision_text}')
                print(f"Decision added: {decision_text}")
                return 0
            except Exception as e:
                print(f"Error adding decision: {e}")
                return 1
        
        elif command == 'status':
            return subprocess.call(['python', '.prsist/prsist.py', '-s'])
        
        elif command == 'health': 
            return subprocess.call(['python', '.prsist/prsist.py', '-h'])
        
        elif command == 'memory':
            return subprocess.call(['python', '.prsist/prsist.py', '-m'])
        
        elif command == 'context':
            return subprocess.call(['python', '.prsist/prsist.py', '-c'])
        
        elif command == 'recent':
            return subprocess.call(['python', '.prsist/prsist.py', '-r'])
        
        else:
            print(f"Unknown command: {command}")
            print("Available: feature, decision, status, health, memory, context, recent")
            return 1
    
    except Exception as e:
        print(f"Command error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(handle_command())