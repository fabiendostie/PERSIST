#!/usr/bin/env node

/**
 * Example: BMAD-Prsist Integration
 * 
 * Demonstrates how to integrate Prsist memory system with BMAD workflows
 * Shows agent memory sharing, story tracking, and architecture decisions
 */

const { BmadPrsistAdapter } = require('../prsist-bridge');

async function demonstrateBmadIntegration() {
    console.log('🚀 BMAD-Prsist Integration Demo\n');

    // Initialize BMAD adapter
    const adapter = new BmadPrsistAdapter({ debug: true });
    await adapter.initialize();

    console.log('✅ BMAD memory adapter initialized\n');

    // 1. Start a development session
    console.log('📅 Starting development session...');
    const session = await adapter.startSession({
        project: 'E-commerce Platform',
        epic: 'User Authentication System'
    });
    console.log(`Session ID: ${session.id}\n`);

    // 2. Simulate analyst decisions
    console.log('👩‍💼 Analyst making decisions...');
    await adapter.captureAgentDecision(
        'analyst',
        'Use JWT tokens for authentication',
        {
            reasoning: 'Better scalability than session-based auth',
            alternatives_considered: ['Session-based', 'OAuth only'],
            confidence: 0.9
        }
    );

    await adapter.captureAgentDecision(
        'analyst',
        'Implement rate limiting for login attempts',
        {
            reasoning: 'Prevent brute force attacks',
            security_requirement: true
        }
    );

    // 3. Simulate architect decisions
    console.log('🏗️  Architect making decisions...');
    await adapter.captureArchitectureDecision(
        'authentication-service',
        'Use Redis for token blacklisting',
        'Fast lookup needed for invalidated tokens'
    );

    await adapter.captureArchitectureDecision(
        'user-database',
        'Separate table for authentication data',
        'Isolate sensitive auth data from user profiles'
    );

    // 4. Simulate story events
    console.log('📝 Tracking story progress...');
    await adapter.captureStoryEvent(
        'User Registration API',
        'created',
        {
            story_points: 5,
            assigned_to: 'dev-agent',
            requirements: ['Input validation', 'Email verification', 'Password hashing']
        }
    );

    await adapter.captureStoryEvent(
        'User Registration API',
        'started',
        {
            start_time: new Date().toISOString(),
            developer: 'dev-agent'
        }
    );

    // 5. Add project memories
    console.log('🧠 Adding project knowledge...');
    await adapter.addProjectMemory(
        'Password hashing: Use bcrypt with cost factor 12 for production',
        'security-guideline'
    );

    await adapter.addDecision(
        'Use PostgreSQL for user data storage',
        'ACID compliance needed for user financial data',
        'high'
    );

    // 6. Create a checkpoint
    console.log('💾 Creating checkpoint...');
    await adapter.createCheckpoint(
        'auth-architecture-complete',
        'Authentication system architecture defined and approved'
    );

    // 7. Simulate git correlation
    console.log('🔗 Correlating with git...');
    await adapter.correlateWithGit(
        'abc123def456',
        'feature/user-authentication'
    );

    // 8. Get BMAD-specific context for next agent
    console.log('🎯 Getting context for development agent...');
    const devContext = await adapter.getBmadContext('dev');
    
    console.log('\n📊 Development Context Summary:');
    console.log(`- Agent decisions: ${devContext.agent_history?.length || 0}`);
    console.log(`- Architecture decisions: ${devContext.bmad_events?.filter(e => e.type === 'architecture_decision').length || 0}`);
    console.log(`- Story events: ${devContext.bmad_events?.filter(e => e.type === 'story_event').length || 0}`);

    // 9. Show memory statistics
    console.log('\n📈 Memory Statistics:');
    const stats = await adapter.getMemoryStats();
    Object.entries(stats).forEach(([key, value]) => {
        console.log(`   ${key}: ${value}`);
    });

    // 10. End session
    console.log('\n🏁 Ending session...');
    await adapter.endSession();

    console.log('\n✅ BMAD-Prsist integration demo completed!');
    console.log('\n💡 This demonstrates how BMAD agents can:');
    console.log('   - Share decisions across the development lifecycle');
    console.log('   - Maintain context between different development phases');
    console.log('   - Track story progress and architecture evolution');
    console.log('   - Build cumulative project knowledge over time');
}

// Example CLI integration
async function demonstrateCliIntegration() {
    console.log('\n🔧 CLI Integration Examples:\n');

    const examples = [
        'prsist agent decision analyst "Use microservices architecture" \'{"confidence": 0.8}\'',
        'prsist story event "User Login API" "completed" \'{"duration_hours": 4}\'',
        'prsist arch "auth-service" "JWT token implementation" "Stateless authentication preferred"',
        'prsist checkpoint "mvp-auth-complete" "Basic authentication system working"',
        'prsist memory "Remember to implement 2FA in phase 2"',
        'prsist agent context dev  # Get development-specific context'
    ];

    console.log('Example CLI commands for BMAD workflows:');
    examples.forEach(example => {
        console.log(`   ${example}`);
    });

    console.log('\n💡 These commands can be integrated into:');
    console.log('   - BMAD agent completion hooks');
    console.log('   - Git commit hooks');
    console.log('   - CI/CD pipeline steps');
    console.log('   - Developer workflow scripts');
}

// Run demo
if (require.main === module) {
    const command = process.argv[2];

    if (command === 'cli') {
        demonstrateCliIntegration();
    } else {
        demonstrateBmadIntegration()
            .then(() => demonstrateCliIntegration())
            .catch(error => {
                console.error('❌ Demo failed:', error.message);
                process.exit(1);
            });
    }
}

module.exports = {
    demonstrateBmadIntegration,
    demonstrateCliIntegration
};