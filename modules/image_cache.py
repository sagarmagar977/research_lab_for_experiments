import os
import io
import time
import base64
import threading
from PIL import Image

# Global thread-safe in-memory cache
# Maps normalized canonical file paths to base64 JPEG strings
_IMAGE_CACHE = {}
_ACTIVE_WORKERS = set()
_CACHE_LOCK = threading.Lock()

def _norm_path(path):
    """Returns normalized lowercase absolute path for collision-free caching across sessions."""
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(path))

def clear_image_cache():
    """Flushes the global in-memory thumbnail cache and active workers."""
    with _CACHE_LOCK:
        _IMAGE_CACHE.clear()
        _ACTIVE_WORKERS.clear()

def _create_thumbnail_b64(path, max_dim=400):
    """Creates a lightweight JPEG thumbnail base64 string from an image path."""
    try:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        except Exception:
            return ""

def get_cached_thumbnail_b64(path, max_dim=400):
    """
    Retrieves thumbnail from in-memory cache or computes it immediately.
    Strictly keyed by absolute canonical path to prevent cross-session filename collisions.
    """
    if not path:
        return ""
        
    key = _norm_path(path)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]
        
    # Generate synchronously if missing (only happens for active page frames)
    thumb = _create_thumbnail_b64(path, max_dim=max_dim)
    if thumb:
        _IMAGE_CACHE[key] = thumb
    return thumb

def get_cache_stats(paths):
    """
    Returns (cached_count, total_count) for a given collection of file paths.
    """
    if not paths:
        return 0, 0
    total = len(paths)
    cached = 0
    for p in paths:
        if _norm_path(p) in _IMAGE_CACHE:
            cached += 1
    return cached, total

def trigger_background_session_prefetch(paths, current_page=1, page_size=24, session_key=None):
    """
    Asynchronously pre-warms all remaining session frames into RAM cache.
    Stage 1: Immediately warms the next page and previous page.
    Stage 2: Progressively warms all remaining frames across all pages.
    Yields CPU (1ms sleep) to ensure main UI rendering is completely uninhibited.
    """
    if not paths:
        return

    # Derive unique worker key based on normalized directory
    if not session_key:
        session_key = _norm_path(os.path.dirname(paths[0])) if paths else "default_session"
    else:
        session_key = _norm_path(session_key)

    with _CACHE_LOCK:
        if session_key in _ACTIVE_WORKERS:
            return  # Worker already active for this session
        _ACTIVE_WORKERS.add(session_key)

    def _worker():
        try:
            # 1. Immediate priority: Next Page (frames user is most likely to click next)
            start_next = current_page * page_size
            end_next = min(start_next + page_size, len(paths))
            if start_next < len(paths):
                for p in paths[start_next:end_next]:
                    k = _norm_path(p)
                    if k not in _IMAGE_CACHE:
                        t = _create_thumbnail_b64(p)
                        if t:
                            _IMAGE_CACHE[k] = t

            # 2. Priority: Previous Page (if user is beyond page 1)
            if current_page > 1:
                start_prev = max(0, (current_page - 2) * page_size)
                end_prev = start_prev + page_size
                for p in paths[start_prev:end_prev]:
                    k = _norm_path(p)
                    if k not in _IMAGE_CACHE:
                        t = _create_thumbnail_b64(p)
                        if t:
                            _IMAGE_CACHE[k] = t

            # 3. Stage 2: Progressively warm all frames of every page in the session
            for p in paths:
                k = _norm_path(p)
                if k not in _IMAGE_CACHE:
                    t = _create_thumbnail_b64(p)
                    if t:
                        _IMAGE_CACHE[k] = t
                # Cooperative yield to prevent CPU starvation
                time.sleep(0.001)

        finally:
            with _CACHE_LOCK:
                _ACTIVE_WORKERS.discard(session_key)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
