# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability within Prsist, please send an email to the project maintainer rather than using the issue tracker.

**Please include the following information in your report:**

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

## Security Features

Prsist includes several built-in security features:

### Input Validation
- All file paths are validated to prevent directory traversal attacks
- User input is sanitized before database operations
- SQL injection prevention through parameterized queries

### File System Security
- Restricted file access to project directory only
- Path canonicalization to prevent access outside permitted areas
- Safe file operations with proper error handling

### Database Security
- SQLite database with atomic transactions
- No sensitive data stored in memory files
- Proper database connection handling and cleanup

### Memory Management
- Limited memory usage (< 50MB)
- Automatic cleanup of temporary data
- No credential storage in memory files

## Security Best Practices for Users

### Repository Security
1. **Enable branch protection** on main branch
2. **Require pull request reviews** for all changes
3. **Enable secret scanning** and push protection
4. **Use Dependabot** for dependency updates
5. **Limit repository collaborators** to trusted users only

### Local Development
1. **Keep Python and dependencies updated**
2. **Use virtual environments** for development
3. **Review code before committing** sensitive changes
4. **Enable git hooks** for automatic validation
5. **Regularly audit system logs** for suspicious activity

### Production Deployment
1. **Use minimal required permissions**
2. **Enable monitoring and logging**
3. **Regularly backup memory databases**
4. **Keep system and dependencies updated**
5. **Monitor for security advisories**

## Known Security Considerations

### Data Storage
- Session data contains conversation history and project context
- Memory files are stored locally and not encrypted at rest
- Database files should be excluded from version control

### Network Security
- No network communication by default
- Git integration uses local git configuration
- Claude Code integration is local-only

### Access Control
- File system access limited to project directory
- No authentication mechanism (relies on system security)
- Memory data accessible to anyone with file system access

## Security Updates

Security updates will be released as patch versions and announced through:
- GitHub releases
- Security advisories (if applicable)
- Repository README updates

## Disclaimer

This software is provided "as is" without warranty of any kind. Users are responsible for:
- Securing their development environment
- Protecting sensitive project data
- Following security best practices
- Regular security audits of their usage

---

For questions about this security policy, please contact the project maintainer.

Last updated: 2025-01-24