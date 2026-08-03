# Security Policy

Pi is a coding agent that runs locally within the security boundary of the user that is running it. It is the responsibility of the user to monitor its operations or to contain it within a container, virtual machine, or other sandbox solution.

Pi treats the local user account and files writable by that account as inside the same trust boundary as the Pi process itself.

## Reporting a Vulnerability

Report vulnerabilities privately:

- Email: `security@earendil.com`
- Or open a private report through GitHub Security Advisories

## Scope

Security issues in the distributed Python packages, command-line tools, APIs, and repository code are in scope.

## Out of Scope

- Local code execution or sandboxing behavior (Pi intentionally does not have a sandbox)
- Behavior of extensions or skills installed by the user
- Risks from working in untrusted repositories
- Prompt injection attacks
- Exposed user-controlled credentials
