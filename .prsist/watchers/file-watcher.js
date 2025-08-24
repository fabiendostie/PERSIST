#!/usr/bin/env node
/**
 * File watcher for Prsist Memory System Phase 3
 * Real-time file monitoring with Chokidar
 */

const chokidar = require('chokidar');
const path = require('path');
const fs = require('fs-extra');
const yaml = require('js-yaml');
const { spawn } = require('child_process');

class ChangeProcessorBatch {
    constructor(batchSize = 50, debounceMs = 300) {
        this.batchSize = batchSize;
        this.debounceMs = debounceMs;
        this.changes = new Map();
        this.debounceTimer = null;
        this.onBatchReady = null;
    }

    addChange(filePath, changeType) {
        // Store change with timestamp
        this.changes.set(filePath, {
            type: changeType,
            timestamp: Date.now(),
            path: filePath
        });

        // Clear existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Set debounce timer
        this.debounceTimer = setTimeout(() => {
            this.processBatch();
        }, this.debounceMs);

        // Process immediately if batch is full
        if (this.changes.size >= this.batchSize) {
            this.processBatch();
        }
    }

    processBatch() {
        if (this.changes.size === 0) return;

        const batch = Array.from(this.changes.values());
        this.changes.clear();

        if (this.onBatchReady) {
            this.onBatchReady(batch);
        }

        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    }
}

class FileWatcher {
    constructor(configPath = '.prsist/config/watch-config.yaml') {
        this.configPath = configPath;
        this.config = null;
        this.watcher = null;
        this.batcher = null;
        this.memoryUpdater = null;
        this.isRunning = false;
    }

    async loadConfig() {
        try {
            if (await fs.pathExists(this.configPath)) {
                const configContent = await fs.readFile(this.configPath, 'utf8');
                this.config = yaml.load(configContent);
            } else {
                // Use default configuration
                this.config = this.getDefaultConfig();
                console.log('Using default watch configuration');
            }
        } catch (error) {
            console.error('Failed to load watch config:', error);
            this.config = this.getDefaultConfig();
        }
    }

    getDefaultConfig() {
        return {
            file_watching: {
                enabled: true,
                watch_paths: [
                    'src/**/*',
                    'docs/**/*',
                    'bmad-core/**/*',
                    '*.{js,ts,py,md,yaml,json}'
                ],
                ignored_patterns: [
                    '/(^|[/\\\\])\\.',  // dotfiles
                    '/node_modules/',
                    '/\\.git/',
                    '/dist|build|output/',
                    '/\\.(log|tmp|cache)$/',
                    '/.prsist/'
                ],
                performance: {
                    debounce_ms: 300,
                    batch_size: 50,
                    max_queue_size: 1000,
                    stability_threshold: 300,
                    poll_interval: 100
                },
                memory_triggers: {
                    code_files: ['js', 'ts', 'py', 'go', 'java', 'cpp', 'cs'],
                    config_files: ['yaml', 'json', 'toml', 'ini'],
                    doc_files: ['md', 'rst', 'txt'],
                    ignore_files: ['log', 'tmp', 'cache', 'lock']
                }
            }
        };
    }

    async start() {
        console.log('Starting Claude Code Memory File Watcher...');

        await this.loadConfig();

        if (!this.config.file_watching?.enabled) {
            console.log('File watching is disabled in configuration');
            return;
        }

        const watchConfig = this.config.file_watching;

        // Initialize batch processor
        this.batcher = new ChangeProcessorBatch(
            watchConfig.performance.batch_size,
            watchConfig.performance.debounce_ms
        );

        this.batcher.onBatchReady = (batch) => {
            this.processBatch(batch);
        };

        // Initialize memory updater
        const { MemoryUpdater } = require('./memory-updater');
        this.memoryUpdater = new MemoryUpdater();

        // Configure chokidar watcher
        const watcherOptions = {
            ignored: watchConfig.ignored_patterns,
            persistent: true,
            ignoreInitial: true,
            awaitWriteFinish: {
                stabilityThreshold: watchConfig.performance.stability_threshold,
                pollInterval: watchConfig.performance.poll_interval
            },
            usePolling: false,
            interval: watchConfig.performance.poll_interval,
            binaryInterval: 300,
            depth: 5,
            atomic: true
        };

        console.log('Watch paths:', watchConfig.watch_paths);
        console.log('Ignored patterns:', watchConfig.ignored_patterns);

        this.watcher = chokidar.watch(watchConfig.watch_paths, watcherOptions);

        // Set up event handlers
        this.watcher
            .on('add', (filePath) => {
                console.log(`File added: ${filePath}`);
                this.batcher.addChange(filePath, 'add');
            })
            .on('change', (filePath) => {
                console.log(`File changed: ${filePath}`);
                this.batcher.addChange(filePath, 'change');
            })
            .on('unlink', (filePath) => {
                console.log(`File deleted: ${filePath}`);
                this.batcher.addChange(filePath, 'delete');
            })
            .on('addDir', (dirPath) => {
                console.log(`Directory added: ${dirPath}`);
                this.batcher.addChange(dirPath, 'add_dir');
            })
            .on('unlinkDir', (dirPath) => {
                console.log(`Directory deleted: ${dirPath}`);
                this.batcher.addChange(dirPath, 'delete_dir');
            })
            .on('ready', () => {
                console.log('File watcher ready. Monitoring for changes...');
                this.isRunning = true;
            })
            .on('error', (error) => {
                console.error('Watcher error:', error);
            });

        // Handle graceful shutdown
        process.on('SIGINT', () => this.stop());
        process.on('SIGTERM', () => this.stop());
    }

    async processBatch(batch) {
        try {
            console.log(`Processing batch of ${batch.length} changes`);

            // Filter changes by file type relevance
            const relevantChanges = this.filterRelevantChanges(batch);
            
            if (relevantChanges.length === 0) {
                console.log('No relevant changes to process');
                return;
            }

            console.log(`${relevantChanges.length} relevant changes found`);

            // Update memory system
            await this.memoryUpdater.processFileChanges(relevantChanges);

            // Trigger change impact analysis
            await this.triggerChangeAnalysis(relevantChanges);

        } catch (error) {
            console.error('Error processing batch:', error);
        }
    }

    filterRelevantChanges(changes) {
        const watchConfig = this.config.file_watching;
        const triggers = watchConfig.memory_triggers;

        return changes.filter(change => {
            const ext = path.extname(change.path).slice(1);
            const fileName = path.basename(change.path);

            // Skip ignored file types
            if (triggers.ignore_files.includes(ext)) {
                return false;
            }

            // Include code files, config files, and documentation
            return (
                triggers.code_files.includes(ext) ||
                triggers.config_files.includes(ext) ||
                triggers.doc_files.includes(ext) ||
                fileName.toLowerCase().includes('readme') ||
                fileName.toLowerCase().includes('changelog')
            );
        });
    }

    async triggerChangeAnalysis(changes) {
        try {
            // Call Python change processor
            const changeProcessor = spawn('python', [
                '.prsist/watchers/change-processor.py',
                JSON.stringify(changes)
            ], {
                cwd: process.cwd(),
                stdio: ['pipe', 'pipe', 'pipe']
            });

            changeProcessor.stdout.on('data', (data) => {
                console.log('Change processor output:', data.toString());
            });

            changeProcessor.stderr.on('data', (data) => {
                console.error('Change processor error:', data.toString());
            });

            changeProcessor.on('close', (code) => {
                if (code !== 0) {
                    console.error(`Change processor exited with code ${code}`);
                }
            });

        } catch (error) {
            console.error('Failed to trigger change analysis:', error);
        }
    }

    async stop() {
        console.log('Stopping file watcher...');
        this.isRunning = false;

        if (this.watcher) {
            await this.watcher.close();
            console.log('File watcher stopped');
        }

        // Process any remaining changes
        if (this.batcher) {
            this.batcher.processBatch();
        }

        process.exit(0);
    }

    getStatus() {
        return {
            running: this.isRunning,
            watchedPaths: this.config?.file_watching?.watch_paths || [],
            queueSize: this.batcher?.changes.size || 0
        };
    }
}

// Main execution
async function main() {
    const watcher = new FileWatcher();
    
    try {
        await watcher.start();
    } catch (error) {
        console.error('Failed to start file watcher:', error);
        process.exit(1);
    }
}

// Export for testing
module.exports = { FileWatcher, ChangeProcessorBatch };

// Run if called directly
if (require.main === module) {
    main();
}