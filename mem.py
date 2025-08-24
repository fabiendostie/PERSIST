#!/usr/bin/env python3
"""
Memory System CLI - Easy access with arguments
Usage: python mem.py <command> [args...]
"""

import sys
import os
import subprocess

# Add .prsist to path
sys.path.insert(0, '.prsist')

def main():
    if len(sys.argv) < 2:
        print("🧠 Memory System Commands:")
        print("  status                           - Show session status")
        print("  health                           - Health check")
        print("  context                          - Show context")
        print("  memory                           - Memory stats")
        print("  test                             - Test system")
        print("  feature <name> <description>     - Log feature completion")
        print("  decision <text>                  - Add decision to memory")
        print("  recent                           - Recent sessions")
        print("")
        print("Examples:")
        print("  python mem.py status")
        print("  python mem.py feature 'API Integration' 'Connected REST API successfully'")
        print("  python mem.py decision 'Use PostgreSQL for better performance'")
        return 1
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    try:
        if command == 'status':
            return subprocess.call(['python', '.prsist/prsist.py', '-s'])
        
        elif command == 'health':
            return subprocess.call(['python', '.prsist/prsist.py', '-h'])
        
        elif command == 'context':
            return subprocess.call(['python', '.prsist/prsist.py', '-c'])
        
        elif command == 'memory':
            return subprocess.call(['python', '.prsist/prsist.py', '-m'])
        
        elif command == 'test':
            return subprocess.call(['python', '.prsist/prsist.py', '-t'])
        
        elif command == 'recent':
            return subprocess.call(['python', '.prsist/prsist.py', '-r'])
        
        elif command == 'feature':
            if len(args) < 2:
                print("Usage: python mem.py feature <name> <description>")
                return 1
            name = args[0]
            description = ' '.join(args[1:])
            return subprocess.call(['python', '.prsist/hooks/FeatureComplete.py', name, description])
        
        elif command == 'decision':
            if len(args) < 1:
                print("Usage: python mem.py decision <decision_text>")
                return 1
            decision_text = ' '.join(args)
            
            # Add decision directly to project memory
            from memory_manager import MemoryManager
            mm = MemoryManager()
            result = mm.add_project_memory(f'**Decision**: {decision_text}')
            print(f"Decision added to project memory: {decision_text}")
            return 0
        
        else:
            print(f"Unknown command: {command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())