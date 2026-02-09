"""LRU tile cache for managing disk-cached tiles."""

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional


class TileCache:
    """Simple LRU cache that tracks tile directories and evicts old ones."""

    def __init__(self, max_entries: int = 1000):
        self._cache: OrderedDict[str, Path] = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str) -> Optional[Path]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, path: Path):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_entries:
                _, old_path = self._cache.popitem(last=False)
                if old_path.exists():
                    shutil.rmtree(old_path, ignore_errors=True)
            self._cache[key] = path

    def invalidate(self, key: str):
        path = self._cache.pop(key, None)
        if path and path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def clear(self):
        for path in self._cache.values():
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        self._cache.clear()
