"""Compatibility namespace for projects migrating to :mod:`fxxkstock`."""

from __future__ import annotations

import fxxkstock as _fxxkstock

__path__ = _fxxkstock.__path__
__all__ = getattr(_fxxkstock, "__all__", [])
