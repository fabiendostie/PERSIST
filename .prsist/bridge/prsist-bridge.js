#!/usr/bin/env node

/**
 * Prsist Memory System - JavaScript Bridge
 * 
 * Provides Node.js/JavaScript interface to Python memory core
 * Enables integration with JavaScript workflows (BMAD, etc.)
 * Maintains workflow-agnostic design for open source distribution
 */

const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

class PrsistBridge {
    constructor(options = {}) {
        this.pythonPath = options.pythonPath || 'python';
        this.prsistRoot = options.prsistRoot || path.resolve(__dirname, '..');
        this.workflowType = options.workflowType || 'generic';
        this.debug = options.debug || false;
        
        this.initialized = false;
    }

    /**
     * Initialize the bridge connection to Python core
     */
    async initialize() {
        try {
            // Test Python core availability
            await this.testPythonCore();
            this.initialized = true;
            
            if (this.debug) {
                console.log('[Prsist Bridge] Initialized successfully');
            }
        } catch (error) {
            throw new Error(`Failed to initialize Prsist bridge: ${error.message}`);
        }
    }

    /**
     * Test if Python core is available and working
     */
    async testPythonCore() {
        return new Promise((resolve, reject) => {
            const testCmd = `${this.pythonPath} ${path.join(this.prsistRoot, 'test_system.py')}`;
            
            exec(testCmd, { cwd: this.prsistRoot }, (error, stdout, stderr) => {
                // Check if test actually passed by looking for success indicators in output
                const output = stdout + stderr;
                if (output.includes('Memory session ended successfully') || 
                    output.includes('All tests passed')) {
                    resolve(output);
                } else if (error) {
                    reject(new Error(`Python core test failed: ${error.message}`));
                } else {
                    reject(new Error(`Python core test failed: ${stderr || 'Unknown error'}`));
                }
            });
        });
    }

    /**
     * Execute Python memory command
     */
    async executeMemoryCommand(command, args = []) {
        if (!this.initialized) {
            await this.initialize();
        }

        return new Promise((resolve, reject) => {
            const pythonScript = path.join(this.prsistRoot, 'prsist.py');
            const pythonArgs = [pythonScript, `-${command}`, ...args];
            
            const pythonProcess = spawn(this.pythonPath, pythonArgs, {
                cwd: this.prsistRoot,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout.trim());
                        resolve(result);
                    } catch (e) {
                        resolve({ output: stdout.trim(), raw: true });
                    }
                } else {
                    reject(new Error(`Memory command failed: ${stderr}`));
                }
            });

            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to execute memory command: ${error.message}`));
            });
        });
    }

    /**
     * Session Management
     */
    async startSession(metadata = {}) {
        const sessionMetadata = {
            workflow: this.workflowType,
            timestamp: new Date().toISOString(),
            ...metadata
        };

        return this.executeMemoryCommand('start_session', [JSON.stringify(sessionMetadata)]);
    }

    async endSession(sessionId = null) {
        const args = sessionId ? [sessionId] : [];
        return this.executeMemoryCommand('end_session', args);
    }

    async createCheckpoint(name, description = '') {
        return this.executeMemoryCommand('create_checkpoint', [name, description]);
    }

    /**
     * Context Management
     */
    async getSessionContext(includeDecisions = true) {
        return this.executeMemoryCommand('get_context', [includeDecisions.toString()]);
    }

    async addProjectMemory(content, type = 'note') {
        return this.executeMemoryCommand('add_memory', [content, type]);
    }

    async addDecision(decision, rationale, impact = 'medium') {
        return this.executeMemoryCommand('add_decision', [decision, rationale, impact]);
    }

    /**
     * Workflow Integration Events
     */
    async captureWorkflowEvent(eventType, eventData) {
        const event = {
            type: eventType,
            workflow: this.workflowType,
            timestamp: new Date().toISOString(),
            data: eventData
        };

        return this.executeMemoryCommand('capture_event', [JSON.stringify(event)]);
    }

    /**
     * Git Integration
     */
    async correlateWithGit(commitHash = null, branchName = null) {
        const gitData = {
            commit: commitHash,
            branch: branchName,
            timestamp: new Date().toISOString()
        };

        return this.executeMemoryCommand('correlate_git', [JSON.stringify(gitData)]);
    }

    /**
     * Analytics and Insights
     */
    async getMemoryStats() {
        return this.executeMemoryCommand('get_stats');
    }

    async getRecentSessions(limit = 10) {
        return this.executeMemoryCommand('get_recent_sessions', [limit.toString()]);
    }

    /**
     * Health and Validation
     */
    async healthCheck() {
        return this.executeMemoryCommand('health_check');
    }

    async validateSystem() {
        return this.executeMemoryCommand('validate');
    }
}

/**
 * BMAD-Specific Workflow Adapter
 */
class BmadPrsistAdapter extends PrsistBridge {
    constructor(options = {}) {
        super({ ...options, workflowType: 'bmad' });
    }

    /**
     * Capture BMAD agent decision
     */
    async captureAgentDecision(agentName, decision, context = {}) {
        const eventData = {
            agent: agentName,
            decision: decision,
            context: context,
            session_type: 'agent_decision'
        };

        return this.captureWorkflowEvent('agent_decision', eventData);
    }

    /**
     * Capture story creation/completion
     */
    async captureStoryEvent(storyTitle, eventType, storyData = {}) {
        const eventData = {
            story_title: storyTitle,
            event_type: eventType, // 'created', 'started', 'completed'
            story_data: storyData
        };

        return this.captureWorkflowEvent('story_event', eventData);
    }

    /**
     * Capture architecture decisions
     */
    async captureArchitectureDecision(component, decision, rationale) {
        const eventData = {
            component: component,
            decision: decision,
            rationale: rationale,
            session_type: 'architecture_decision'
        };

        return this.captureWorkflowEvent('architecture_decision', eventData);
    }

    /**
     * Get BMAD-specific context for agents
     */
    async getBmadContext(agentType = null) {
        const context = await this.getSessionContext();
        
        // Filter for BMAD-relevant context
        if (context && context.workflow_events) {
            context.bmad_events = context.workflow_events.filter(
                event => event.workflow === 'bmad'
            );
        }

        if (agentType) {
            // Filter for agent-specific context
            context.agent_history = context.bmad_events?.filter(
                event => event.data.agent === agentType
            ) || [];
        }

        return context;
    }
}

/**
 * Factory function for creating workflow-specific adapters
 */
function createPrsistAdapter(workflowType, options = {}) {
    switch (workflowType.toLowerCase()) {
        case 'bmad':
            return new BmadPrsistAdapter(options);
        case 'generic':
        default:
            return new PrsistBridge({ ...options, workflowType });
    }
}

module.exports = {
    PrsistBridge,
    BmadPrsistAdapter,
    createPrsistAdapter
};

// CLI interface when run directly
if (require.main === module) {
    const command = process.argv[2];
    const args = process.argv.slice(3);

    const bridge = new PrsistBridge({ debug: true });

    switch (command) {
        case 'test':
            bridge.testPythonCore()
                .then(() => console.log('✅ Prsist bridge test successful'))
                .catch(err => {
                    console.error('❌ Prsist bridge test failed:', err.message);
                    process.exit(1);
                });
            break;

        case 'health':
            bridge.healthCheck()
                .then(result => console.log('Health:', result))
                .catch(err => console.error('Health check failed:', err.message));
            break;

        case 'context':
            bridge.getSessionContext()
                .then(context => console.log(JSON.stringify(context, null, 2)))
                .catch(err => console.error('Failed to get context:', err.message));
            break;

        default:
            console.log('Prsist JavaScript Bridge');
            console.log('Usage: node prsist-bridge.js <command>');
            console.log('Commands: test, health, context');
    }
}