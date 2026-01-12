import os
import importlib
import pkgutil
import inspect
from typing import Dict, Type

from .base import TKBCModel

# Model registry: {class_name -> class}
ALL_MODELS: Dict[str, Type[TKBCModel]] = {}

# Scan sibling modules under this package and auto-register TKBCModel subclasses
_package_dir = os.path.dirname(__file__)

for _, _module_name, _ in pkgutil.iter_modules([_package_dir]):
    # Skip non-model modules
    if _module_name in {"base", "components"}:
        continue

    _module = importlib.import_module(f".{_module_name}", package=__package__)

    # Register all TKBCModel subclasses found in the module
    for _name, _obj in inspect.getmembers(_module, inspect.isclass):
        if issubclass(_obj, TKBCModel) and _obj is not TKBCModel:
            ALL_MODELS[_name] = _obj


def get_model_class(model_name: str) -> Type[TKBCModel]:
    """Return the registered model class by name."""
    try:
        return ALL_MODELS[model_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {sorted(ALL_MODELS.keys())}"
        ) from e
