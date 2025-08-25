#!/usr/bin/env python3
"""
Prsist Memory System Installer
Smart installer that sets up memory persistence for any project while respecting Claude Code conventions.

Usage:
    python install-prsist.py                    # Install in current directory
    python install-prsist.py --target /path     # Install in specific directory
    python install-prsist.py --uninstall        # Remove Prsist from current directory
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path

class PrsistInstaller:
    def __init__(self, target_dir=None):
        self.target_dir = Path(target_dir) if target_dir else Path.cwd()
        self.source_dir = Path(__file__).parent
        
    def install(self):
        """Install Prsist memory system in target directory"""
        print(f"🧠 Installing Prsist Memory System in {self.target_dir}")
        
        # Check if target directory exists
        if not self.target_dir.exists():
            print(f"❌ Target directory does not exist: {self.target_dir}")
            return False
            
        # Create necessary directories
        self._create_directories()
        
        # Copy core system
        self._copy_prsist_core()
        
        # Copy Claude Code integration (respecting .claude/ location)
        self._setup_claude_integration()
        
        # Copy convenience CLI scripts
        self._copy_cli_scripts()
        
        # Copy git hooks configuration
        self._copy_git_hooks()
        
        # Update .gitignore
        self._update_gitignore()
        
        # Initialize system
        self._initialize_system()
        
        print("✅ Prsist Memory System installed successfully!")
        print("\nQuick start:")
        print("  python mem.py status          # Check system status")
        print("  python mem.py health          # Run health check")
        print("  /mem-status                   # Claude Code command")
        
        return True
    
    def uninstall(self):
        """Remove Prsist memory system from target directory"""
        print(f"🗑️  Uninstalling Prsist Memory System from {self.target_dir}")
        
        files_to_remove = [
            ".prsist",
            ".claude/agents/memory-manager.md",
            ".lefthook.yml",
            "mem.py", 
            "memory-cli.py",
            "claude-commands.py"
        ]
        
        # Remove Claude commands
        claude_commands_dir = self.target_dir / ".claude" / "commands"
        if claude_commands_dir.exists():
            for cmd_file in claude_commands_dir.glob("mem-*.md"):
                cmd_file.unlink()
                print(f"  Removed: {cmd_file}")
        
        # Remove main files
        for file_path in files_to_remove:
            full_path = self.target_dir / file_path
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()
                print(f"  Removed: {full_path}")
        
        print("✅ Prsist Memory System uninstalled successfully!")
        return True
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.target_dir / ".claude" / "agents",
            self.target_dir / ".claude" / "commands"
        ]
        
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _copy_prsist_core(self):
        """Copy the entire .prsist directory"""
        source_prsist = self.source_dir / ".prsist"
        target_prsist = self.target_dir / ".prsist"
        
        if source_prsist.exists():
            if target_prsist.exists():
                shutil.rmtree(target_prsist)
            shutil.copytree(source_prsist, target_prsist)
            print(f"  Copied: .prsist/ ({self._count_files(target_prsist)} files)")
        else:
            print("❌ Source .prsist directory not found")
            return False
    
    def _setup_claude_integration(self):
        """Setup Claude Code integration files"""
        # Copy memory manager agent
        source_agent = self.source_dir / ".claude" / "agents" / "memory-manager.md"
        target_agent = self.target_dir / ".claude" / "agents" / "memory-manager.md"
        
        if source_agent.exists():
            shutil.copy2(source_agent, target_agent)
            print(f"  Copied: .claude/agents/memory-manager.md")
        
        # Copy all mem-*.md commands
        source_commands = self.source_dir / ".claude" / "commands"
        target_commands = self.target_dir / ".claude" / "commands"
        
        if source_commands.exists():
            mem_commands = list(source_commands.glob("mem-*.md"))
            for cmd_file in mem_commands:
                target_file = target_commands / cmd_file.name
                shutil.copy2(cmd_file, target_file)
            print(f"  Copied: {len(mem_commands)} Claude Code commands")
    
    def _copy_cli_scripts(self):
        """Copy convenience CLI scripts to root"""
        cli_scripts = ["mem.py", "memory-cli.py", "claude-commands.py"]
        
        for script in cli_scripts:
            source_file = self.source_dir / script
            target_file = self.target_dir / script
            
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                # Make executable on Unix systems
                if os.name != 'nt':
                    os.chmod(target_file, 0o755)
                print(f"  Copied: {script}")
    
    def _copy_git_hooks(self):
        """Copy git hooks configuration"""
        source_hooks = self.source_dir / ".lefthook.yml"
        target_hooks = self.target_dir / ".lefthook.yml"
        
        if source_hooks.exists():
            if target_hooks.exists():
                print("  .lefthook.yml already exists, creating backup")
                shutil.copy2(target_hooks, target_hooks.with_suffix('.yml.backup'))
            shutil.copy2(source_hooks, target_hooks)
            print(f"  Copied: .lefthook.yml")
    
    def _update_gitignore(self):
        """Update .gitignore to include/exclude appropriate files"""
        gitignore_path = self.target_dir / ".gitignore"
        
        lines_to_add = [
            "# Prsist Memory System",
            ".prsist/sessions/",
            ".prsist/storage/",
            ".prsist/logs/",
            ".prsist/tests/test_results.json",
            ""
        ]
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            if "# Prsist Memory System" not in content:
                with open(gitignore_path, 'a') as f:
                    f.write('\n' + '\n'.join(lines_to_add))
                print("  Updated: .gitignore")
        else:
            with open(gitignore_path, 'w') as f:
                f.write('\n'.join(lines_to_add))
            print("  Created: .gitignore")
    
    def _initialize_system(self):
        """Initialize the memory system"""
        init_script = self.target_dir / ".prsist" / "bin" / "prsist.py"
        if init_script.exists():
            os.chdir(self.target_dir)
            os.system(f'python "{init_script}" --init')
    
    def _count_files(self, directory):
        """Count files in directory recursively"""
        return sum(1 for _ in directory.rglob('*') if _.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Install Prsist Memory System for persistent AI memory"
    )
    parser.add_argument(
        '--target', 
        help="Target directory (default: current directory)"
    )
    parser.add_argument(
        '--uninstall', 
        action='store_true',
        help="Uninstall Prsist from target directory"
    )
    
    args = parser.parse_args()
    
    installer = PrsistInstaller(args.target)
    
    if args.uninstall:
        success = installer.uninstall()
    else:
        success = installer.install()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()