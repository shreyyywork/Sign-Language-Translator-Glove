"""
Windows Smart App Control / Code Integrity compatibility shim for scikit-learn.
On Windows with Python 3.14, sklearn 1.9.0's compiled histogram.cp314-win_amd64.pyd
(used only by HistGradientBoosting) may be blocked by Windows Application Control (WinError 4551).
Since this project uses DecisionTreeClassifier, RandomForestClassifier, and SVM (none of which use histogram.pyd),
we inject a lightweight dummy into sys.modules before sklearn.ensemble imports to avoid loading the blocked binary.
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
