# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Price-Prophet, please **do not** open a public GitHub issue.

Instead, email **devneatharva@gmail.com** with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (optional)

We will respond within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

In scope:
- API injection vulnerabilities
- Authentication/authorization bypasses
- SQL injection in database queries
- Deserialization vulnerabilities in model loading
- Sensitive data exposure

Out of scope:
- Denial-of-service attacks requiring excessive resources
- Social engineering
- Issues in third-party dependencies (report upstream)

## Security Considerations

- Model files (`*.pkl`) should never be loaded from untrusted sources — pickle deserialization is inherently unsafe
- The `DATABASE_URL` environment variable may contain credentials; never log it
- Restrict `/train` endpoint in production to authorized clients only
