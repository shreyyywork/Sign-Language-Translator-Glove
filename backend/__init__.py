"""
Backend package initialization with Windows Application Control compatibility shim.
"""
import sys
import types

_mod_name = 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting'
if _mod_name not in sys.modules:
    _shim = types.ModuleType(_mod_name)
    class _HistDummy:
        pass
    _shim.HistGradientBoostingClassifier = _HistDummy
    _shim.HistGradientBoostingRegressor = _HistDummy
    sys.modules[_mod_name] = _shim
