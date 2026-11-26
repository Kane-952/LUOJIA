"""Capacity policy for LangGraph's in-process cache."""
from collections import OrderedDict
from threading import Lock

from langgraph.cache.memory import InMemoryCache


class BoundedGraphCache:
    def __init__(self, namespace, capacity, *, ttl=None):
        self.namespace = namespace
        self.capacity = capacity
        self.ttl = ttl
        self.backend = InMemoryCache()
        self.keys = OrderedDict()
        self.lock = Lock()

    def _namespace(self, key):
        return (self.namespace, key)

    def get(self, key):
        with self.lock:
            if key not in self.keys:
                return None
            full_key = (self._namespace(key), key)
            value = self.backend.get([full_key]).get(full_key)
            if value is None:
                self.keys.pop(key)
                self.backend.clear([full_key[0]])
            else:
                self.keys.move_to_end(key)
            return value

    def set(self, key, value):
        if self.capacity <= 0:
            return
        with self.lock:
            self.backend.set({(self._namespace(key), key): (value, self.ttl)})
            self.keys[key] = None
            self.keys.move_to_end(key)
            if len(self.keys) > self.capacity:
                oldest, _ = self.keys.popitem(last=False)
                self.backend.clear([self._namespace(oldest)])

    def discard(self, key):
        with self.lock:
            self.keys.pop(key, None)
            self.backend.clear([self._namespace(key)])

    def clear(self):
        with self.lock:
            self.keys.clear()
            self.backend.clear()
