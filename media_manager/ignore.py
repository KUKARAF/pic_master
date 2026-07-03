"""
Read .mediaignore (git-ignore format) and offer is_ignored(path).
"""
import os
import pathlib
import fnmatch

class IgnoreRules:
    """
    Very small git-ignore style matcher.
    Accepts patterns like:
        *.jpg
        cache/
        **/temp*
    """
    def __init__(self, repo_root):
        self.repo_root = pathlib.Path(repo_root).resolve()
        self.patterns = []
        ignore_file = self.repo_root / '.mediaignore'
        if ignore_file.is_file():
            with ignore_file.open(encoding='utf-8') as fh:
                self.patterns = [line.rstrip() for line in fh if line.strip() and not line.startswith('#')]

    def is_ignored(self, relative_path):
        """
        relative_path: str  (relative to repo root, POSIX separators)
        Returns True when the file should be ignored.
        """
        parts = relative_path.split('/')
        # full-path pattern match
        for pat in self.patterns:
            if pat.startswith('**/'):
                # leading **/  →  match anywhere below root
                stem = pat[3:]
                for part in parts:
                    if fnmatch.fnmatch(part, stem) or fnmatch.fnmatch(relative_path, stem):
                        return True
            elif '/' in pat:
                # pat contains /  →  match against relative_path
                if fnmatch.fnmatch(relative_path, pat):
                    return True
            else:
                # basename or last part match
                if fnmatch.fnmatch(parts[-1], pat):
                    return True
                for part in parts[:-1]:
                    if fnmatch.fnmatch(part, pat):
                        return True
        return False
