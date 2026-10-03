"""Bootstrap helpers for the test suite.

The shipped test classes call ``self.assertIsNotNone`` without inheriting
from ``unittest.TestCase``. Supply the assertion helpers to such classes at
collection time without touching the test files themselves.
"""

import unittest


_ASSERTION_HELPERS = (
    "fail",
    "_formatMessage",
    "_getAssertEqualityFunc",
)
_ASSERTION_ATTRS = (
    "failureException",
    "longMessage",
    "maxDiff",
    "_diffThreshold",
    "_type_equality_funcs",
)


def pytest_collection_modifyitems(items):
    patched = set()
    for item in items:
        cls = getattr(item, "cls", None)
        if cls is None or cls in patched or issubclass(cls, unittest.TestCase):
            continue
        patched.add(cls)
        for attr in _ASSERTION_ATTRS:
            if not hasattr(cls, attr) and hasattr(unittest.TestCase, attr):
                setattr(cls, attr, getattr(unittest.TestCase, attr))
        for method_name in dir(unittest.TestCase):
            if method_name.startswith(("assert", "assertNot", "assertEqual")) \
                    and not hasattr(cls, method_name):
                setattr(cls, method_name, getattr(unittest.TestCase, method_name))
        for helper in _ASSERTION_HELPERS:
            if not hasattr(cls, helper):
                setattr(cls, helper, getattr(unittest.TestCase, helper))
