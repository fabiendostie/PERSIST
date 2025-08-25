# Force Context Injection

**Command**: `/mem-force-context`

**Description**: Emergency failsafe to force inject project context and memory when automatic context injection fails or when you need to refresh context mid-session.

**Usage**: Use this command when:
- Context seems missing or stale
- Automatic context injection failed during session start
- You need to refresh project memory mid-conversation
- Memory system appears disconnected

**Command**:
```bash
python .prsist/bin/prsist.py -i
```

**Fallback**:
If the Python command fails, manually read the context files:
- Read: .prsist/context/claude-context.md
- Read: .prsist/sessions/active/current-session.json

**Expected Output**:
- Current project context
- Recent session information
- Project memory and decisions
- Memory system status

**Use Cases**:
- Session context appears missing
- Need to reload project memory
- Memory system status unclear
- Context injection hook failed