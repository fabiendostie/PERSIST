#!/usr/bin/env node

/**
 * Simple CLI test for Prsist Bridge
 */

const { PrsistBridge } = require('./prsist-bridge');

async function testBridge() {
    const bridge = new PrsistBridge({ debug: false });
    
    console.log('🧠 Prsist Memory System - Bridge Test');
    console.log('=====================================');
    
    try {
        // Test health check
        console.log('Testing health check...');
        const health = await bridge.executeMemoryCommand('h', []);
        console.log('✅ Health check passed');
        
        // Test status
        console.log('Testing status...');
        const status = await bridge.executeMemoryCommand('s', []);
        console.log('✅ Status check passed');
        
        // Test memory stats
        console.log('Testing memory stats...');
        const stats = await bridge.executeMemoryCommand('m', []);
        console.log('✅ Memory stats passed');
        
        console.log('\n🎉 All bridge tests passed!');
        console.log('JavaScript-Python communication is working correctly');
        
    } catch (error) {
        console.error('❌ Bridge test failed:', error.message || error);
        process.exit(1);
    }
}

if (require.main === module) {
    testBridge();
}