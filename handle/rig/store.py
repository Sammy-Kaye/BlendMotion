"""Persist a mapping on the armature object as a JSON custom property.

The property survives save/load of the .blend and is not an extra rig layer
(it is just metadata on the object).
"""
import json

from .mapper import RigMapping

PROP = "handle_mapping"
_cache = {}   # (object name, json text) -> RigMapping


def store_mapping(obj, mapping: RigMapping):
    obj[PROP] = json.dumps(mapping.to_dict())


def fetch_mapping(obj):
    """Stored mapping for `obj`, or None."""
    text = obj.get(PROP) if obj is not None else None
    if not text:
        return None
    key = (obj.name, text)
    m = _cache.get(key)
    if m is None:
        if len(_cache) > 32:
            _cache.clear()
        try:
            m = RigMapping.from_dict(json.loads(text))
        except (ValueError, KeyError):
            return None
        _cache[key] = m
    return m


def clear_mapping(obj):
    if obj is not None and PROP in obj:
        del obj[PROP]
