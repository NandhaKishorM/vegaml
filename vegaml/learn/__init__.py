"""Continuous learning moved to its own package, vegaml-learn, in vegaml 0.20.

vegaml-learn is licensed under the PolyForm Noncommercial License 1.0.0: free for noncommercial use;
commercial use needs a licence from Embedding Space (https://embedding.in). vegaml itself stays
Apache 2.0. Install it with `pip install vegaml-learn` (`vegaml-learn[serve]` for the REST server,
`[proxy]` for the vega Proxy, `[claude]` for claude_teacher).

Once it is installed, `vegaml.learn` and its submodules (`vegaml.learn.proxy`, ...) are the same
module objects as `vegaml_learn` and its submodules, so existing imports keep working.
"""

import importlib
import importlib.abc
import importlib.util
import sys

MOVED = (
    "Continuous learning (Learner, Escalator, the vega Proxy) moved to its own package in vegaml 0.20. "
    "Install it with `pip install vegaml-learn`. It is free for noncommercial use (PolyForm Noncommercial "
    "1.0.0); commercial use needs a licence from Embedding Space, https://embedding.in."
)

try:
    import vegaml_learn as _impl
except ModuleNotFoundError as exc:
    if exc.name != "vegaml_learn":
        raise
    raise ModuleNotFoundError(MOVED, name="vegaml_learn") from None


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`vegaml.learn.X` resolves to the already-imported `vegaml_learn.X`, never a second copy."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(__name__ + ".") and fullname != __name__ + ".__main__":
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_impl.__name__ + spec.name[len(__name__):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.append(_Alias())

__all__ = list(_impl.__all__)
globals().update({name: getattr(_impl, name) for name in __all__})
