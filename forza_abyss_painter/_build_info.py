"""Build provenance — overwritten by CI before PyInstaller runs.

In a dev checkout this file ships its placeholder values. The release
workflow (.github/workflows/release.yml) rewrites this file with the
real commit SHA + build timestamp + tag immediately before the
PyInstaller step, so the bundled EXE knows what commit it came from.
Surfaced in the About dialog so users + bug reporters can identify
which build they're running without digging through git.
"""
from __future__ import annotations

# Short git SHA the EXE was built from. "dev" means a local non-CI build.
BUILD_SHA: str = "dev"

# Git tag the build was published under (eg "v1.0.0"). Empty in non-release
# builds.
BUILD_TAG: str = ""

# UTC timestamp of the CI build (ISO 8601). Empty in non-release builds.
BUILD_TIMESTAMP: str = ""
