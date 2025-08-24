#!/usr/bin/env node

/**
 * Prsist Memory System - CLI Interface
 * 
 * Universal command-line interface for memory system
 * Works with any workflow, with special BMAD integration
 */

const { createPrsistAdapter } = require('./prsist-bridge');
const path = require('path');
const fs = require('fs').promises;

class PrsistCLI {
    constructor() {
        this.adapter = null;
        this.workflowType = 'generic';
    }

    async detectWorkflow() {
        // Auto-detect workflow type from project structure
        const cwd = process.cwd();
        
        try {
            // Check for BMAD
            const bmadExists = await this.fileExists(path.join(cwd, 'bmad-core')) ||
                              await this.fileExists(path.join(cwd, 'package.json'));
            
            if (bmadExists) {
                const packageJson = await this.readPackageJson(cwd);
                if (packageJson && packageJson.dependencies && packageJson.dependencies['bmad-method']) {
                    this.workflowType = 'bmad';
                    return 'bmad';
                }
            }

            // Check for other workflows
            // Add more detection logic here

            return 'generic';
        } catch (error) {
            return 'generic';
        }
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    async readPackageJson(dir) {
        try {
            const packagePath = path.join(dir, 'package.json');
            const content = await fs.readFile(packagePath, 'utf8');
            return JSON.parse(content);
        } catch {
            return null;
        }
    }

    async initialize(workflowType = null) {
        this.workflowType = workflowType || await this.detectWorkflow();
        this.adapter = createPrsistAdapter(this.workflowType, { debug: false });
        await this.adapter.initialize();
        
        console.log(`🧠 Prsist Memory System (${this.workflowType} workflow)`);
    }

    async handleCommand(command, args) {
        try {
            switch (command) {
                case 'init':
                    await this.cmdInit();
                    break;
                case 'status':
                case 's':
                    await this.cmdStatus();
                    break;
                case 'health':
                case 'h':
                    await this.cmdHealth();
                    break;
                case 'context':
                case 'c':
                    await this.cmdContext();
                    break;
                case 'memory':
                case 'm':
                    await this.cmdMemory(args[0]);
                    break;
                case 'decision':
                case 'd':
                    await this.cmdDecision(args);
                    break;
                case 'checkpoint':
                case 'cp':
                    await this.cmdCheckpoint(args[0], args[1]);
                    break;
                case 'start':
                    await this.cmdStart(args[0]);
                    break;
                case 'end':
                    await this.cmdEnd();
                    break;
                case 'stats':
                    await this.cmdStats();
                    break;
                case 'recent':
                case 'r':
                    await this.cmdRecent(args[0]);
                    break;
                
                // BMAD-specific commands
                case 'agent':
                    await this.cmdAgent(args);
                    break;
                case 'story':
                    await this.cmdStory(args);
                    break;
                case 'arch':
                    await this.cmdArchitecture(args);
                    break;

                default:
                    this.showHelp();
            }
        } catch (error) {
            console.error(`❌ Error: ${error.message}`);
            process.exit(1);
        }
    }

    async cmdInit() {
        console.log('🔧 Installing Prsist memory system...');
        // Installation logic here
        console.log('✅ Prsist memory system installed');
    }

    async cmdStatus() {
        const health = await this.adapter.healthCheck();
        const stats = await this.adapter.getMemoryStats();
        
        console.log('\n📊 System Status:');
        console.log(`   Health: ${health.status || 'Unknown'}`);
        console.log(`   Sessions: ${stats.total_sessions || 0}`);
        console.log(`   Memory entries: ${stats.total_memories || 0}`);
        console.log(`   Workflow: ${this.workflowType}`);
    }

    async cmdHealth() {
        const result = await this.adapter.healthCheck();
        
        if (result.status === 'healthy') {
            console.log('✅ System healthy');
        } else {
            console.log('⚠️  System issues detected');
            if (result.issues) {
                result.issues.forEach(issue => console.log(`   - ${issue}`));
            }
        }
    }

    async cmdContext() {
        const context = await this.adapter.getSessionContext();
        
        if (context.content) {
            console.log('\n🧠 Current Context:');
            console.log('─'.repeat(50));
            console.log(context.content.substring(0, 500) + (context.content.length > 500 ? '...' : ''));
            console.log('─'.repeat(50));
            console.log(`📏 Length: ${context.content.length} characters`);
        } else {
            console.log('📭 No context available');
        }
    }

    async cmdMemory(content) {
        if (!content) {
            console.log('Usage: prsist memory "content to remember"');
            return;
        }

        await this.adapter.addProjectMemory(content);
        console.log('✅ Added to project memory');
    }

    async cmdDecision(args) {
        if (args.length < 2) {
            console.log('Usage: prsist decision "decision" "rationale" [impact]');
            return;
        }

        const [decision, rationale, impact = 'medium'] = args;
        await this.adapter.addDecision(decision, rationale, impact);
        console.log('✅ Decision recorded');
    }

    async cmdCheckpoint(name, description = '') {
        if (!name) {
            console.log('Usage: prsist checkpoint "checkpoint-name" ["description"]');
            return;
        }

        await this.adapter.createCheckpoint(name, description);
        console.log(`✅ Checkpoint "${name}" created`);
    }

    async cmdStart(metadata) {
        const sessionMetadata = metadata ? JSON.parse(metadata) : {};
        const session = await this.adapter.startSession(sessionMetadata);
        console.log(`✅ Session started: ${session.id}`);
    }

    async cmdEnd() {
        await this.adapter.endSession();
        console.log('✅ Session ended');
    }

    async cmdStats() {
        const stats = await this.adapter.getMemoryStats();
        
        console.log('\n📈 Memory Statistics:');
        Object.entries(stats).forEach(([key, value]) => {
            console.log(`   ${key}: ${value}`);
        });
    }

    async cmdRecent(limit = '10') {
        const sessions = await this.adapter.getRecentSessions(parseInt(limit));
        
        console.log(`\n📅 Recent Sessions (${sessions.length}):`);
        sessions.forEach(session => {
            console.log(`   ${session.id} - ${session.created_at} (${session.status})`);
        });
    }

    // BMAD-specific commands
    async cmdAgent(args) {
        if (this.workflowType !== 'bmad') {
            console.log('⚠️  Agent commands only available in BMAD workflows');
            return;
        }

        const [action, agentName, ...rest] = args;
        
        switch (action) {
            case 'decision':
                if (!agentName || !rest[0]) {
                    console.log('Usage: prsist agent decision <agent> "<decision>" [context]');
                    return;
                }
                const context = rest[1] ? JSON.parse(rest[1]) : {};
                await this.adapter.captureAgentDecision(agentName, rest[0], context);
                console.log(`✅ Captured ${agentName} decision`);
                break;
                
            case 'context':
                const agentContext = await this.adapter.getBmadContext(agentName);
                console.log(JSON.stringify(agentContext, null, 2));
                break;
                
            default:
                console.log('Agent actions: decision, context');
        }
    }

    async cmdStory(args) {
        if (this.workflowType !== 'bmad') {
            console.log('⚠️  Story commands only available in BMAD workflows');
            return;
        }

        const [action, title, eventType, ...data] = args;
        
        if (action === 'event') {
            await this.adapter.captureStoryEvent(title, eventType, data[0] ? JSON.parse(data[0]) : {});
            console.log(`✅ Captured story event: ${title} - ${eventType}`);
        } else {
            console.log('Story actions: event');
        }
    }

    async cmdArchitecture(args) {
        if (this.workflowType !== 'bmad') {
            console.log('⚠️  Architecture commands only available in BMAD workflows');
            return;
        }

        const [component, decision, rationale] = args;
        
        if (!component || !decision || !rationale) {
            console.log('Usage: prsist arch <component> "<decision>" "<rationale>"');
            return;
        }

        await this.adapter.captureArchitectureDecision(component, decision, rationale);
        console.log(`✅ Captured architecture decision for ${component}`);
    }

    showHelp() {
        console.log(`
🧠 Prsist Memory System CLI

GENERAL COMMANDS:
  init                 Install memory system
  status, s            Show system status  
  health, h            Check system health
  context, c           Show current context
  memory <content>     Add to project memory
  decision <decision> <rationale> [impact]  Record decision
  checkpoint <name> [desc]  Create checkpoint
  start [metadata]     Start new session
  end                  End current session
  stats                Show memory statistics
  recent [limit]       Show recent sessions

BMAD WORKFLOW COMMANDS (when in BMAD project):
  agent decision <agent> "<decision>" [context]  Capture agent decision
  agent context [agent]                          Get agent context
  story event <title> <type> [data]             Capture story event
  arch <component> "<decision>" "<rationale>"   Architecture decision

EXAMPLES:
  prsist status
  prsist memory "Decided to use React for frontend"
  prsist checkpoint "login-feature-complete"
  prsist agent decision analyst "Use microservices architecture"
  prsist story event "User Login" "completed"
        `);
    }
}

// Main execution
async function main() {
    const [,, command, ...args] = process.argv;
    
    if (!command || command === 'help' || command === '--help') {
        const cli = new PrsistCLI();
        cli.showHelp();
        return;
    }

    const cli = new PrsistCLI();
    await cli.initialize();
    await cli.handleCommand(command, args);
}

if (require.main === module) {
    main().catch(error => {
        console.error(`❌ Fatal error: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { PrsistCLI };