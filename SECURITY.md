# Security Policy

FD6 is a desktop tool that can read and write memory in a running Forza Horizon
process for vinyl-group injection. Treat all builds as local tools and avoid
running binaries from untrusted sources.

## Reporting Issues

Please open a GitHub security advisory or contact the maintainers privately if
you find a vulnerability that could:

- Execute code unexpectedly.
- Write outside the selected game process.
- Corrupt files or saved data.
- Leak local paths, user data, or generated artwork.

For normal bugs, crashes, game-version offset changes, or generation-quality
problems, open a standard GitHub issue.

## Supported Versions

Only the current `main` branch and the latest GitHub release are supported for
security fixes.
