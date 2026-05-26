r"""Resolve per-title livery folders inside the user's Documents directory.

ACC + ACE + AC Rally all store custom liveries under:
  %USERPROFILE%\Documents\<title-folder>\Customs\Liveries\

Original AC is different — liveries live inside the install directory,
not Documents — so it's handled separately when that title is implemented.

Functions in this module are read-only inspectors. They never create folders
on their own; the livery_writer module owns folder creation. If a title's
livery root doesn't exist (game not installed, or Documents structure
not initialized), inspectors return None and the GUI surfaces a
'(no install detected)' message instead of erroring.
"""

from __future__ import annotations

import os
from pathlib import Path

from fd6.ac.profiles import ACTitleProfile


def _documents_root() -> Path | None:
    """Resolve %USERPROFILE%\\Documents — the parent of every AC user folder.

    Honors a redirected Documents location if the user has moved it via
    Windows's folder redirection. Returns None if neither env var is
    available (non-Windows, broken environment, etc.).
    """
    # USERPROFILE is set on every modern Windows install. Documents is
    # almost always at USERPROFILE\Documents unless the user has redirected
    # it; we honor that redirect via the SHGetKnownFolderPath fallback below.
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        docs = Path(user_profile) / "Documents"
        if docs.exists():
            return docs

    # Fallback: ctypes SHGetKnownFolderPath for FOLDERID_Documents. This
    # picks up redirected Documents locations the env-var path misses.
    try:
        import ctypes
        from ctypes import wintypes
        FOLDERID_Documents = ctypes.c_char_p(b"\xfd\xf2\xfe\xfd\xe9\x67\xe1\x4b\xa1\xc6\xea\x8d\x21\x8c\xff\x7e")
        # Simpler path: use the documented CSIDL_PERSONAL via SHGetFolderPathW
        # which is still supported on Windows 10/11 and avoids GUID juggling.
        CSIDL_PERSONAL = 5
        SHGFP_TYPE_CURRENT = 0
        buf = ctypes.create_unicode_buffer(260)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        result = shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf)
        if result == 0 and buf.value:
            p = Path(buf.value)
            if p.exists():
                return p
    except Exception:
        pass
    return None


def livery_root(profile: ACTitleProfile) -> Path | None:
    """Return the Customs\\Liveries folder for the given title, or None if it
    can't be located (Documents folder missing or title's subpath absent).

    Does NOT create the folder. Callers ('livery_writer' specifically) create
    sub-team folders inside the livery root when they need to write.
    """
    if not profile.user_folder_subpath:
        # Original AC — livery folder lives inside the install dir, not here.
        return None
    docs = _documents_root()
    if docs is None:
        return None
    candidate = docs.joinpath(*profile.user_folder_subpath)
    if candidate.exists():
        return candidate
    return None


def is_installed(profile: ACTitleProfile) -> bool:
    """True if the title's livery root is present on disk.

    Detecting 'installed' purely via Documents is a heuristic — a user could
    in theory have the game installed without ever opening the livery editor,
    in which case the folder won't exist yet. The Export button in the GUI
    will create the folder structure on first export anyway; this function
    is for the up-front 'looks like the game is set up' check.
    """
    return livery_root(profile) is not None
