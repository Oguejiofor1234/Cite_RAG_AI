import os
import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal and unsafe characters.
    Extracts only the base filename and strips dangerous characters.
    """
    if not filename:
        return "document.pdf"
    
    # Normalize path separators (both POSIX and Windows)
    normalized = filename.replace("\\", "/")
    basename = os.path.basename(normalized)
    
    # Strip any consecutive dots to eliminate any residual traversal components
    basename = re.sub(r'\.{2,}', '', basename)
    
    # Replace dangerous characters while preserving alphanumeric, dashes, underscores, and dots
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', basename)
    
    # Ensure it ends with .pdf
    if not sanitized.lower().endswith(".pdf"):
        sanitized = f"{sanitized}.pdf"
        
    # Prevent hidden files or leading underscore/dots
    sanitized = sanitized.lstrip("._")
    if not sanitized:
        sanitized = "document.pdf"
        
    return sanitized


def validate_pdf_path(path: str, allowed_dirs: list[str]) -> bool:
    """
    Validate that a given file path is strictly located within one of the allowed directories.
    Prevents path traversal attacks attempting to read arbitrary files from the filesystem.
    """
    try:
        target_path = Path(path).resolve()
        for allowed in allowed_dirs:
            allowed_path = Path(allowed).resolve()
            if target_path == allowed_path or allowed_path in target_path.parents:
                return True
        return False
    except Exception:
        return False
