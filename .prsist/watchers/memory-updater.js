/**
 * Memory updater for Prsist Memory System Phase 3
 * Handles memory updates from file changes
 */

const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');

class MemoryUpdater {
    constructor() {
        this.updateQueue = [];
        this.isProcessing = false;
        this.maxQueueSize = 1000;
    }

    async processFileChanges(changes) {
        console.log(`Memory updater processing ${changes.length} file changes`);

        try {
            // Analyze changes for memory relevance
            const memoryRelevantChanges = await this.analyzeMemoryRelevance(changes);

            // Queue updates
            for (const change of memoryRelevantChanges) {
                await this.queueMemoryUpdate(change);
            }

            // Process queue
            await this.processUpdateQueue();

        } catch (error) {
            console.error('Error processing file changes:', error);
        }
    }

    async analyzeMemoryRelevance(changes) {
        const relevantChanges = [];

        for (const change of changes) {
            const relevance = await this.calculateFileRelevance(change);
            
            if (relevance.score > 0.3) {  // Minimum relevance threshold
                relevantChanges.push({
                    ...change,
                    relevance: relevance
                });
            }
        }

        return relevantChanges;
    }

    async calculateFileRelevance(change) {
        const filePath = change.path;
        const ext = path.extname(filePath).slice(1);
        const fileName = path.basename(filePath);
        
        let score = 0.0;
        let factors = [];

        // File type scoring
        const fileTypeScores = {
            'py': 0.8, 'js': 0.8, 'ts': 0.8, 'go': 0.7, 'java': 0.7,
            'cpp': 0.7, 'cs': 0.7, 'rs': 0.7, 'rb': 0.6, 'php': 0.6,
            'md': 0.6, 'rst': 0.5, 'txt': 0.3,
            'yaml': 0.7, 'json': 0.6, 'toml': 0.5, 'ini': 0.4,
            'dockerfile': 0.6, 'makefile': 0.5
        };

        if (fileTypeScores[ext.toLowerCase()]) {
            score += fileTypeScores[ext.toLowerCase()];
            factors.push(`file_type:${ext}`);
        }

        // Special file name patterns
        const specialPatterns = {
            'readme': 0.8, 'changelog': 0.7, 'todo': 0.6,
            'config': 0.6, 'package': 0.6, 'requirements': 0.7,
            'test': 0.5, 'spec': 0.5
        };

        for (const [pattern, patternScore] of Object.entries(specialPatterns)) {
            if (fileName.toLowerCase().includes(pattern)) {
                score += patternScore;
                factors.push(`pattern:${pattern}`);
            }
        }

        // Directory context scoring
        const pathParts = filePath.split(path.sep);
        const directoryScores = {
            'src': 0.8, 'lib': 0.7, 'app': 0.7, 'core': 0.8,
            'bmad-core': 0.9, 'docs': 0.6, 'config': 0.6,
            'test': 0.4, 'tests': 0.4, 'spec': 0.4,
            'node_modules': 0.0, 'build': 0.1, 'dist': 0.1
        };

        for (const part of pathParts) {
            if (directoryScores[part.toLowerCase()] !== undefined) {
                score += directoryScores[part.toLowerCase()];
                factors.push(`dir:${part}`);
            }
        }

        // Change type impact
        const changeTypeScores = {
            'add': 0.8,
            'change': 0.6,
            'delete': 0.4,
            'add_dir': 0.3,
            'delete_dir': 0.2
        };

        if (changeTypeScores[change.type]) {
            score += changeTypeScores[change.type];
            factors.push(`change:${change.type}`);
        }

        // File size consideration (if available)
        try {
            if (change.type !== 'delete' && await fs.pathExists(filePath)) {
                const stats = await fs.stat(filePath);
                const sizeKB = stats.size / 1024;
                
                if (sizeKB > 0 && sizeKB < 1000) {  // 0-1MB files are most relevant
                    score += Math.min(sizeKB / 1000, 0.3);
                    factors.push(`size:${sizeKB.toFixed(1)}KB`);
                }
            }
        } catch (error) {
            // File might not exist or be accessible
        }

        return {
            score: Math.min(score, 1.0),
            factors: factors,
            timestamp: Date.now()
        };
    }

    async queueMemoryUpdate(change) {
        if (this.updateQueue.length >= this.maxQueueSize) {
            console.warn('Memory update queue is full, dropping oldest updates');
            this.updateQueue.shift();
        }

        const update = {
            id: `update_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            change: change,
            timestamp: Date.now(),
            status: 'queued'
        };

        this.updateQueue.push(update);
        console.log(`Queued memory update: ${change.path} (relevance: ${change.relevance.score.toFixed(2)})`);
    }

    async processUpdateQueue() {
        if (this.isProcessing || this.updateQueue.length === 0) {
            return;
        }

        this.isProcessing = true;
        console.log(`Processing ${this.updateQueue.length} memory updates`);

        try {
            while (this.updateQueue.length > 0) {
                const update = this.updateQueue.shift();
                await this.processUpdate(update);
            }
        } catch (error) {
            console.error('Error processing update queue:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    async processUpdate(update) {
        try {
            update.status = 'processing';
            
            // Call Python memory system to update memory
            await this.callMemorySystem(update);
            
            update.status = 'completed';
            console.log(`Completed memory update: ${update.change.path}`);

        } catch (error) {
            update.status = 'failed';
            update.error = error.message;
            console.error(`Failed memory update: ${update.change.path}:`, error);
        }
    }

    async callMemorySystem(update) {
        return new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [
                '.prsist/memory_manager.py',
                'update_from_file_change',
                JSON.stringify(update)
            ], {
                cwd: process.cwd(),
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
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (parseError) {
                        resolve({ success: true, output: stdout });
                    }
                } else {
                    reject(new Error(`Python process failed (code ${code}): ${stderr}`));
                }
            });

            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to spawn Python process: ${error.message}`));
            });
        });
    }

    getQueueStatus() {
        return {
            queueSize: this.updateQueue.length,
            isProcessing: this.isProcessing,
            maxQueueSize: this.maxQueueSize,
            updates: this.updateQueue.map(u => ({
                id: u.id,
                filePath: u.change.path,
                status: u.status,
                relevance: u.change.relevance.score,
                timestamp: u.timestamp
            }))
        };
    }

    async clearQueue() {
        this.updateQueue = [];
        console.log('Memory update queue cleared');
    }

    async stop() {
        console.log('Stopping memory updater...');
        
        // Process remaining updates
        if (this.updateQueue.length > 0) {
            console.log(`Processing ${this.updateQueue.length} remaining updates`);
            await this.processUpdateQueue();
        }
        
        console.log('Memory updater stopped');
    }
}

module.exports = { MemoryUpdater };