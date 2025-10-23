from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import datetime as _dt
import re
import copy
import yaml

# ==== YAML helpers ====

def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML must be a mapping (dict). File: {path}")
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e

def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """ Recursively merge `updates` into `base` (dicts only). Returns a NEW dict."""
    out = copy.deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(out[k], v)  # type: ignore[index]
        else:
            out[k] = copy.deepcopy(v)
    return out

# ==== Overrides ====

def _coerce_scalar(text: str) -> Any:
    """ Parse scalar/list/dict from a string override.
    Usa YAML loader para soportar: ints, floats, bools, null, listas, dicts.
    """
    try:
        return yaml.safe_load(text)
    except Exception:
        return text

def _parse_dot_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """ Convert --set a.b.c=42 --set renderer.backend=opencv → nested dict."""
    tree: Dict[str, Any] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override (missing '='): {raw!r}")
        path, value = raw.split("=", 1)
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Invalid override path: {raw!r}")
        node = tree
        for k in keys[:-1]:
            node = node.setdefault(k, {})  # type: ignore[assignment]
            if not isinstance(node, dict):
                raise ValueError(f"Override path conflicts with non-dict at {k} in {raw!r}")
        node[keys[-1]] = _coerce_scalar(value.strip())
    return tree

# ==== Substitutions ====
_SUB_RE = re.compile(r"\$\{([^}]+)\}")

def _lookup_path(cfg: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur

def _expand_substitutions_in_str(cfg: Mapping[str, Any], s: str) -> str:
    """ Replace ${a.b.c} inside a string using values from the config mapping."""
    def _repl(m: re.Match[str]) -> str:
        path = m.group(1).strip()
        try:
            val = _lookup_path(cfg, path)
        except KeyError:
            return m.group(0)
        return str(val)
    return _SUB_RE.sub(_repl, s)

def _expand_substitutions(cfg: Any, root: Mapping[str, Any]) -> Any:
    """Recursively expand ${...} in all string leaves of cfg, using root for lookups."""
    if isinstance(cfg, str):
        return _expand_substitutions_in_str(root, cfg)
    elif isinstance(cfg, (list, tuple)):
        return type(cfg)(_expand_substitutions(v, root) for v in cfg)
    elif isinstance(cfg, dict):
        return {k: _expand_substitutions(v, root) for k, v in cfg.items()}
    return cfg

# ==== Public API ====

def load_config(config_path: str, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    """
    Loads a YAML configuration, and applies CLI overrides.

    Args:
    - config_path: path to the base YAML (e.g., `configs/base.yaml`).
    Also accepts an absolute path to a YAML file.
    - overrides: list of strings in `a.b.c=value` notation. The value is parsed with YAML.

    Returns:
    - dict with the final configuration. Applies:
      - merge (base <- overrides)
      - `YYYYMMDD-HHMMSS` timestamp appended to `experiment.output_dir` (if present)
      - expansion of placeholders `${...}` (e.g., `${experiment.name}`)
    """
    base_path = Path(config_path).expanduser().resolve()
    base_cfg = _read_yaml(base_path)
    merged = base_cfg
    if overrides:
        ov = _parse_dot_overrides(overrides)
        merged = _deep_update(merged, ov)

    try:
        exp = merged.get("experiment", {})
        out_dir = exp.get("output_dir")
        if isinstance(out_dir, str) and out_dir:
            # Expand any variables in output_dir first
            out_dir = _expand_substitutions_in_str(merged, out_dir)
            # Then append timestamp
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            merged["experiment"]["output_dir"] = str(Path(out_dir) / ts)
    except Exception as e:
        logging.warning("Failed to process 'experiment.output_dir': %s", e)

    merged = _expand_substitutions(merged, merged)

    return merged