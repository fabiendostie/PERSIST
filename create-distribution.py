#!/usr/bin/env python3
"""
Create Prsist Distribution Package
Creates a portable, self-contained distribution of the Prsist Memory System.

Usage:
    python create-distribution.py                    # Create dist in ./prsist-dist/
    python create-distribution.py --output /path     # Create dist in specific location
    python create-distribution.py --zip              # Create ZIP archive
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

class PrsistDistributor:
    def __init__(self, output_dir=None, create_zip=False):
        self.source_dir = Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else Path("prsist-dist")
        self.create_zip = create_zip
        self.version = self._get_version()
        
    def create_distribution(self):
        """Create complete distribution package"""
        print(f"📦 Creating Prsist Memory System v{self.version} distribution")
        print(f"   Source: {self.source_dir}")
        print(f"   Output: {self.output_dir}")
        
        # Clean existing output
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        # Create distribution structure
        self.output_dir.mkdir(parents=True)
        
        # Copy installer
        self._copy_installer()
        
        # Copy core system
        self._copy_prsist_core()
        
        # Copy Claude integration
        self._copy_claude_integration()
        
        # Copy CLI scripts
        self._copy_cli_scripts()
        
        # Copy configuration files
        self._copy_config_files()
        
        # Create documentation
        self._create_documentation()
        
        # Create version info
        self._create_version_info()
        
        # Create ZIP if requested
        if self.create_zip:
            self._create_zip_archive()
        
        print("✅ Distribution created successfully!")
        print(f"\nDistribution contents:")
        self._show_distribution_contents()
        
        return True
    
    def _get_version(self):
        """Get version from git or default"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'describe', '--tags', '--always'], 
                capture_output=True, text=True, cwd=self.source_dir
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return datetime.now().strftime("dev-%Y%m%d-%H%M")
    
    def _copy_installer(self):
        """Copy the installer script"""
        source = self.source_dir / "install-prsist.py"
        target = self.output_dir / "install.py"
        
        if source.exists():
            shutil.copy2(source, target)
            # Make executable
            if os.name != 'nt':
                os.chmod(target, 0o755)
            print(f"  ✓ Installer: install.py")
        else:
            print("❌ Installer script not found")
            return False
    
    def _copy_prsist_core(self):
        """Copy the complete .prsist directory"""
        source = self.source_dir / ".prsist"
        target = self.output_dir / ".prsist"
        
        if source.exists():
            # Copy entire directory but exclude session data and logs
            shutil.copytree(source, target, ignore=self._ignore_runtime_files)
            file_count = sum(1 for _ in target.rglob('*') if _.is_file())
            print(f"  ✓ Core system: .prsist/ ({file_count} files)")
        else:
            print("❌ Core .prsist directory not found")
            return False
    
    def _copy_claude_integration(self):
        """Copy Claude Code integration files"""
        # Create .claude directory structure
        claude_dir = self.output_dir / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "agents").mkdir(exist_ok=True)
        (claude_dir / "commands").mkdir(exist_ok=True)
        
        # Copy memory manager agent
        source_agent = self.source_dir / ".claude" / "agents" / "memory-manager.md"
        target_agent = claude_dir / "agents" / "memory-manager.md"
        
        if source_agent.exists():
            shutil.copy2(source_agent, target_agent)
            print(f"  ✓ Agent: memory-manager.md")
        
        # Copy all mem-*.md commands
        source_commands = self.source_dir / ".claude" / "commands"
        target_commands = claude_dir / "commands"
        
        if source_commands.exists():
            mem_commands = list(source_commands.glob("mem-*.md"))
            for cmd_file in mem_commands:
                target_file = target_commands / cmd_file.name
                shutil.copy2(cmd_file, target_file)
            print(f"  ✓ Commands: {len(mem_commands)} mem-*.md files")
    
    def _copy_cli_scripts(self):
        """Copy convenience CLI scripts"""
        cli_scripts = ["mem.py", "memory-cli.py", "claude-commands.py"]
        
        for script in cli_scripts:
            source = self.source_dir / script
            target = self.output_dir / script
            
            if source.exists():
                shutil.copy2(source, target)
                if os.name != 'nt':
                    os.chmod(target, 0o755)
                print(f"  ✓ CLI: {script}")
    
    def _copy_config_files(self):
        """Copy configuration files"""
        config_files = [".lefthook.yml"]
        
        for config_file in config_files:
            source = self.source_dir / config_file
            target = self.output_dir / config_file
            
            if source.exists():
                shutil.copy2(source, target)
                print(f"  ✓ Config: {config_file}")
    
    def _create_documentation(self):
        """Create installation and usage documentation"""
        readme_content = f"""# Prsist Memory System v{self.version}

Persistent memory system for AI conversations and project development.

## Quick Installation

```bash
python install.py
```

This will install Prsist in your current project directory with:
- `.prsist/` - Core memory system
- `.claude/` - Claude Code integration  
- CLI scripts for easy access
- Git hooks for automatic tracking

## Usage

### Command Line Interface
```bash
python mem.py status          # Check system status
python mem.py health          # Run health check
python mem.py context         # Show current context
python mem.py memory          # Memory statistics
python mem.py recent          # Recent sessions
```

### Claude Code Integration
```bash
/mem-status                   # System status
/mem-context                  # Current context
/mem-memory                   # Memory stats
/mem-feature "Name" "Desc"    # Log completed feature
/mem-decision "Decision text" # Add decision to memory
```

### Advanced Usage
```bash
python memory-cli.py feature "API Integration" "REST API completed"
python claude-commands.py decision "Use PostgreSQL for performance"
```

## Uninstallation

```bash
python install.py --uninstall
```

## System Requirements

- Python 3.7+
- Git (optional, for advanced features)
- Claude Code (optional, for AI integration)

## What Gets Installed

- **Core System**: `.prsist/` directory with memory management
- **Claude Integration**: `.claude/agents/` and `.claude/commands/` 
- **CLI Tools**: `mem.py`, `memory-cli.py`, `claude-commands.py`
- **Git Hooks**: `.lefthook.yml` for automatic memory updates
- **Configuration**: Updated `.gitignore` with appropriate exclusions

## Features

- ✅ Persistent conversation memory across sessions
- ✅ Project memory for long-term decisions and context
- ✅ Automatic context updates via git hooks
- ✅ Claude Code integration with slash commands
- ✅ Session checkpoints and feature tracking
- ✅ Health monitoring and validation
- ✅ Export capabilities for session data

## Support

For issues, questions, or contributions, visit the project repository.
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"  ✓ Documentation: README.md")
    
    def _create_version_info(self):
        """Create version information file"""
        version_info = {
            "version": self.version,
            "build_date": datetime.now().isoformat(),
            "components": {
                "core_system": ".prsist/",
                "claude_integration": ".claude/",
                "cli_tools": ["mem.py", "memory-cli.py", "claude-commands.py"],
                "configuration": [".lefthook.yml"]
            }
        }
        
        import json
        version_path = self.output_dir / "version.json"
        with open(version_path, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2)
        print(f"  ✓ Version: v{self.version}")
    
    def _ignore_runtime_files(self, directory, files):
        """Ignore runtime files that shouldn't be distributed"""
        ignore = []
        
        # Ignore session data and logs
        if 'sessions' in str(directory):
            ignore.extend([f for f in files if f.endswith('.json')])
        
        if 'logs' in str(directory):
            ignore.extend([f for f in files if f.endswith('.log')])
        
        if 'storage' in str(directory):
            ignore.extend([f for f in files if f.endswith('.log')])
        
        # Ignore test results
        if 'test_results.json' in files:
            ignore.append('test_results.json')
        
        # Ignore corrupted files
        ignore.extend([f for f in files if 'corrupted' in f])
        
        return ignore
    
    def _create_zip_archive(self):
        """Create ZIP archive of the distribution"""
        zip_path = self.output_dir.parent / f"prsist-v{self.version}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file():
                    arc_path = file_path.relative_to(self.output_dir.parent)
                    zipf.write(file_path, arc_path)
        
        print(f"  ✓ Archive: {zip_path.name}")
        return zip_path
    
    def _show_distribution_contents(self):
        """Show what's in the distribution"""
        for item in sorted(self.output_dir.iterdir()):
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                print(f"   📁 {item.name}/ ({file_count} files)")
            else:
                print(f"   📄 {item.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Prsist Memory System distribution package"
    )
    parser.add_argument(
        '--output', 
        help="Output directory (default: ./prsist-dist/)"
    )
    parser.add_argument(
        '--zip', 
        action='store_true',
        help="Create ZIP archive in addition to directory"
    )
    
    args = parser.parse_args()
    
    distributor = PrsistDistributor(args.output, args.zip)
    success = distributor.create_distribution()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()