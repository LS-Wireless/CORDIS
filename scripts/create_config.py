"""
scripts/create_config.py
========================
Create a new CORDIS experiment config file from the default, with
validated overrides specified on the command line or from a partial
override JSON.

Why not copy-paste JSON manually?
----------------------------------
Manual JSON editing has two failure modes: silent typos (wrong key names
that get ignored by the loader) and invalid values that only surface at
runtime deep inside a simulation.  This script validates every override
immediately and shows a clear diff so you always know exactly what
changed from the default.

Workflows
---------

1. Override individual parameters on the command line:
   python scripts/create_config.py \\
       --name exp_scalability \\
       --set topology.n_ap=12 topology.n_ue=8 channel.snr_db=25 \\
       --set algorithm.admm.kappa=2.0

2. Write a small override JSON (only the fields that differ) and merge it:
   python scripts/create_config.py \\
       --name exp_imperfect_csi \\
       --override-json configs/overrides/imperfect_csi.json

3. Both at once (JSON overrides applied first, then CLI --set flags):
   python scripts/create_config.py \\
       --name exp_custom \\
       --override-json configs/overrides/base_override.json \\
       --set simulation.n_trials=1000

4. Preview what would be created without saving:
   python scripts/create_config.py \\
       --name exp_preview \\
       --set topology.n_ap=8 \\
       --dry-run

Override JSON format
--------------------
Use the same nested structure as default.json but include only the fields
you want to change.  Everything else comes from the default:

    {
      "topology": { "n_ap": 12, "n_ue": 8 },
      "channel":  { "snr_db": 25.0 }
    }

--set syntax
------------
Use dot-separated paths matching the JSON structure:
    topology.n_ap=12
    algorithm.admm.kappa=2.0
    channel.estimation_method=perfect
    sensing.los_probability_override=0.9

Values are automatically cast to the correct Python type based on the
dataclass field definition (int, float, bool, str, None/null).
"""

from __future__ import annotations

SCRIPT_DESCRIPTION = (
    "Create a new CORDIS experiment config from the default with validated "
    "overrides.  Shows a diff of what changed and saves the result to configs/."
)

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cordis.utils.config import (
    load_config, config_from_dict, CORDISConfig,
    _deep_merge, PARAM_REGISTRY,
)
from cordis.utils.logger import setup_logging, get_logger
from cordis.utils.paths import project_path

logger = get_logger("create_config")


# =============================================================================
# Type coercion
# =============================================================================

def _coerce(value_str: str, target_type: Any) -> Any:
    """
    Cast a CLI string value to the type expected by the dataclass field.

    Handles: int, float, bool, str, Optional[X], None/null.
    """
    # Unwrap Optional[X] → X
    import typing
    origin = getattr(target_type, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in target_type.__args__ if a is not type(None)]
        if value_str.lower() in ("null", "none"):
            return None
        target_type = args[0] if args else str

    if value_str.lower() in ("null", "none"):
        return None
    if target_type is bool or target_type == "bool":
        if value_str.lower() in ("true", "1", "yes"):
            return True
        if value_str.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Cannot parse '{value_str}' as bool")
    if target_type is int:
        return int(value_str)
    if target_type is float:
        return float(value_str)
    return value_str   # str — no conversion needed


def _resolve_field_type(path: str) -> Optional[Any]:
    """
    Look up the Python type of a config field by its dot-separated path.
    Returns None if the path is not found.

    Field types are stored as strings when from __future__ import annotations
    is active, so we resolve them explicitly here.
    """
    from cordis.utils.config import (
        TopologyConfig, FrequencyConfig, ChannelConfig, SensingConfig,
        ADMMConfig, SplitOptConfig, SimulationConfig,
    )

    _TYPE_MAP = {"str": str, "int": int, "float": float, "bool": bool,
                 "None": type(None)}

    SECTION_MAP = {
        "topology":        TopologyConfig,
        "frequency":       FrequencyConfig,
        "channel":         ChannelConfig,
        "sensing":         SensingConfig,
        "algorithm.admm":  ADMMConfig,
        "algorithm.split": SplitOptConfig,
        "simulation":      SimulationConfig,
    }

    parts = path.split(".")
    # Try two-level section keys first (algorithm.admm, algorithm.split)
    for depth in (2, 1):
        section_key = ".".join(parts[:depth])
        field_name  = ".".join(parts[depth:])
        if section_key in SECTION_MAP and field_name:
            cls = SECTION_MAP[section_key]
            for f in dataclasses.fields(cls):
                if f.name == field_name:
                    raw = f.type
                    # Resolve string annotation to actual type
                    if isinstance(raw, str):
                        # Handle "Optional[float]" -> float, "int" -> int, etc.
                        raw = raw.strip()
                        if raw.startswith("Optional["):
                            inner = raw[9:-1]
                            return _TYPE_MAP.get(inner, str)
                        return _TYPE_MAP.get(raw, str)
                    return raw
    return None


# =============================================================================
# Override application
# =============================================================================

def _set_nested(d: Dict, path: str, value: Any) -> None:
    """Set d[key1][key2]... = value using a dot-separated path."""
    keys = path.split(".")
    node = d
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_set_flags(base_dict: Dict, set_flags: List[str]) -> Dict:
    """
    Parse and apply a list of 'path=value' strings onto base_dict.
    Validates paths against PARAM_REGISTRY and coerces types.
    """
    result = copy.deepcopy(base_dict)

    for flag in set_flags:
        if "=" not in flag:
            raise ValueError(
                f"Invalid --set value '{flag}'. "
                f"Expected format: path.to.field=value"
            )
        path, value_str = flag.split("=", 1)
        path = path.strip()
        value_str = value_str.strip()

        # Warn if path is not in registry (may still be valid)
        if path not in PARAM_REGISTRY:
            logger.warning(
                "Unknown parameter path '%s'. "
                "Check 'python scripts/list_config_params.py' for valid paths.",
                path,
            )

        # Coerce value to correct type
        target_type = _resolve_field_type(path)
        if target_type is not None:
            try:
                value = _coerce(value_str, target_type)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot set '{path}': could not convert '{value_str}' "
                    f"to {target_type}: {e}"
                )
        else:
            # Fallback: try int → float → str
            value: Any = value_str
            for cast in (int, float):
                try:
                    value = cast(value_str)
                    break
                except ValueError:
                    pass
            if value_str.lower() == "true":  value = True
            if value_str.lower() == "false": value = False
            if value_str.lower() in ("null", "none"): value = None

        _set_nested(result, path, value)
        logger.debug("Set %s = %r (type: %s)", path, value,
                     type(value).__name__)

    return result


# =============================================================================
# Diff computation
# =============================================================================

def _flat_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict to dot-separated keys."""
    result = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flat_dict(v, full_key))
        else:
            result[full_key] = v
    return result


def compute_diff(
    default_dict: Dict,
    new_dict: Dict,
) -> List[tuple[str, Any, Any]]:
    """
    Return a list of (path, old_value, new_value) tuples for every field
    that changed from default to new.
    """
    flat_default = _flat_dict(default_dict)
    flat_new     = _flat_dict(new_dict)

    changes = []
    for key in sorted(set(flat_default) | set(flat_new)):
        old = flat_default.get(key, "<missing>")
        new = flat_new.get(key, "<missing>")
        if old != new:
            changes.append((key, old, new))
    return changes


def print_diff(changes: List[tuple], cfg_name: str) -> None:
    """Print a human-readable diff."""
    BOLD  = "\033[1m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    DIM   = "\033[2m"
    RESET = "\033[0m"
    SEP   = "─" * 72

    print()
    print(BOLD + "  Changes from default.json → " + cfg_name + ".json" + RESET)
    print(DIM + "  " + SEP + RESET)

    if not changes:
        print("  (no changes — identical to default)")
    else:
        col = max(len(p) for p, *_ in changes) + 2
        for path, old, new in changes:
            info = PARAM_REGISTRY.get(path, {})
            unit = info.get("unit", "")
            unit_str = f"  [{unit}]" if unit and unit != "-" else ""
            print(
                f"  {path:<{col}} "
                + RED   + f"{old}" + RESET
                + " → "
                + GREEN + f"{new}" + RESET
                + DIM   + unit_str + RESET
            )
    print(DIM + "  " + SEP + RESET)
    print(f"  {len(changes)} field(s) changed.")
    print()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="create_config.py",
        description=SCRIPT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Scalability experiment: more APs and UEs
  python scripts/create_config.py --name exp_scalability \\
      --set topology.n_ap=12 topology.n_ue=8

  # Imperfect CSI study: change estimation method and pilot length
  python scripts/create_config.py --name exp_imperfect_csi \\
      --set channel.estimation_method=MMSE channel.tau_p=4 channel.pilot_power_db=10

  # Clutter sensitivity: vary CNR and clutter angular spread
  python scripts/create_config.py --name exp_clutter \\
      --set sensing.clutter_cnr_db=-5 sensing.clutter_as_deg=25

  # Merge from a partial override JSON, then add CLI overrides
  python scripts/create_config.py --name exp_custom \\
      --override-json configs/overrides/base.json \\
      --set simulation.n_trials=1000 simulation.seed=99

  # Preview without saving
  python scripts/create_config.py --name exp_preview \\
      --set topology.n_ap=20 --dry-run

  # Save to a custom directory
  python scripts/create_config.py --name mmwave_scenario \\
      --set frequency.carrier_freq_ghz=28 \\
      --out-dir configs/mmwave
""",
    )
    parser.add_argument(
        "--name", required=True, metavar="NAME",
        help="Config name (without .json extension); file saved as configs/<NAME>.json",
    )
    parser.add_argument(
        "--base", default="configs/default.json", metavar="PATH",
        help="Base config to start from (default: configs/default.json)",
    )
    parser.add_argument(
        "--override-json", default=None, metavar="PATH",
        help="Optional partial override JSON; only fields present here override the base",
    )
    parser.add_argument(
        "--set", nargs="+", default=[], metavar="path=value",
        help=(
            "One or more 'path=value' overrides applied after --override-json. "
            "Use dot-separated paths matching the JSON structure "
            "(e.g. topology.n_ap=12, algorithm.admm.kappa=2.0)"
        ),
    )
    parser.add_argument(
        "--out-dir", default="configs", metavar="PATH",
        help="Directory in which to save the new config (default: configs/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the diff and validate but do not write any file",
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Skip printing the diff (useful in scripts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    # ── Load base config as a plain dict ──────────────────────────────────
    base_path = project_path(*args.base.split("/"))
    if not base_path.exists():
        logger.error("Base config not found: %s", base_path)
        sys.exit(1)

    with open(base_path, encoding="utf-8") as f:
        base_dict: Dict = json.load(f)

    working_dict = copy.deepcopy(base_dict)

    # ── Apply override JSON ───────────────────────────────────────────────
    if args.override_json:
        override_path = project_path(*args.override_json.split("/"))
        if not override_path.exists():
            logger.error("Override JSON not found: %s", override_path)
            sys.exit(1)
        with open(override_path, encoding="utf-8") as f:
            override_dict = json.load(f)
        working_dict = _deep_merge(working_dict, override_dict)
        logger.info("Applied override JSON: %s", override_path)

    # ── Apply --set flags ─────────────────────────────────────────────────
    if args.set:
        try:
            working_dict = apply_set_flags(working_dict, args.set)
        except ValueError as e:
            logger.error("%s", e)
            sys.exit(1)

    # ── Validate the result ───────────────────────────────────────────────
    try:
        cfg = config_from_dict(working_dict)
    except (ValueError, TypeError) as e:
        logger.error("Validation failed: %s", e)
        sys.exit(1)

    logger.info("Validation passed.")

    # ── Show diff ─────────────────────────────────────────────────────────
    if not args.no_diff:
        changes = compute_diff(base_dict, working_dict)
        print_diff(changes, args.name)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir  = project_path(*args.out_dir.split("/"))
    out_path = out_dir / f"{args.name}.json"

    if args.dry_run:
        logger.info("--dry-run: file not written (would save to %s)", out_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.warning("Overwriting existing config: %s", out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(working_dict, f, indent=2)

    logger.info("Config saved → %s", out_path)
    print(f"  ✓  configs/{args.name}.json  ({len(changes)} changes from default)")


if __name__ == "__main__":
    main()

