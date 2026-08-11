"""
tests/test_config.py
====================
The CORDIS parameter contract.

`configs/default.json` is the single source of truth at runtime; the dataclass
tree in `cordis/utils/config.py` mirrors it, `PARAM_REGISTRY` documents it, and
`configs/recipes/_defaults.sh` re-states it for the shell recipes. Four copies
of the same information, so these tests exist to keep them from drifting.

Known drifts are listed in the `KNOWN_*` registries below and marked
`xfail(strict=True)`, so that fixing one turns into a loud failure that says
"now delete the entry". Each registry names the finding that tracks it.
"""
from __future__ import annotations

import dataclasses as dc
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from cordis.utils.config import (
    CORDISConfig, PARAM_REGISTRY, config_from_dict, load_config, _deep_merge,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "configs" / "default.json"
RECIPE_DEFAULTS = REPO_ROOT / "configs" / "recipes" / "_defaults.sh"


# =============================================================================
# Known drifts (each entry cites the finding that tracks it)
# =============================================================================

#: dotted path -> (dataclass default, default.json value). F-01-01.
KNOWN_DEFAULT_MISMATCHES: dict[str, tuple[Any, Any]] = {
    "algorithm.admm.kappa":              (0.1, 0.08),
    "algorithm.gamma_db":                (10.0, 5.0),
    "channel.pilot_power_db":            (120.0, 128.0),
    "sensing.clutter_center_strategy":   ("target_centroid", "offset"),
    "simulation.n_trials":               (500, 100),
    "topology.n_ant":                    (10, 16),
    "topology.n_ap":                     (6, 10),
    "topology.n_rf_chains":              (10, 16),
    "topology.n_targets":                (1, 2),
}

#: fields with no PARAM_REGISTRY entry. F-01-02.
KNOWN_UNDOCUMENTED: set[str] = {
    "channel.environment",
    "simulation.benchmarks",
    "simulation.log_file",
    "simulation.metrics",
    "simulation.progress_bar",
}


# =============================================================================
# Helpers
# =============================================================================

def _walk_fields(cls, prefix: str = "") -> dict[str, dc.Field]:
    """dotted path -> dataclasses.Field, recursing into nested config classes."""
    out: dict[str, dc.Field] = {}
    for f in dc.fields(cls):
        ftype = f.type
        if isinstance(ftype, str):
            try:
                ftype = eval(ftype, dict(vars(sys.modules[cls.__module__])))
            except Exception:
                pass
        path = f"{prefix}{f.name}"
        if dc.is_dataclass(ftype):
            out.update(_walk_fields(ftype, prefix=f"{path}."))
        else:
            out[path] = f
    return out


def _field_default(f: dc.Field) -> Any:
    if f.default is not dc.MISSING:
        return f.default
    if f.default_factory is not dc.MISSING:      # type: ignore[misc]
        return f.default_factory()               # type: ignore[misc]
    raise AssertionError(f"field {f.name} has no default")


def _walk_json(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_walk_json(v, prefix=f"{path}."))
        else:
            out[path] = v
    return out


FIELDS = _walk_fields(CORDISConfig)
JSON_VALUES = _walk_json(json.loads(DEFAULT_JSON.read_text()))


def _param(path: str, known: dict | set):
    marks = ([pytest.mark.xfail(strict=True,
                                reason=f"known drift, tracked as a Stage-1 finding: {path}")]
             if path in known else [])
    return pytest.param(path, id=path, marks=marks)


# =============================================================================
# 1. Dataclass defaults vs configs/default.json
# =============================================================================

def test_field_sets_are_identical() -> None:
    """
    The dataclass tree and `configs/default.json` describe the same field set.

    Property: no field is declared without a JSON entry (which would make its
    value invisible to the config system), and no JSON key lacks a field (which
    would be silently dropped by `_populate_dataclass`).
    """
    only_dc = sorted(set(FIELDS) - set(JSON_VALUES))
    only_js = sorted(set(JSON_VALUES) - set(FIELDS))
    assert not only_dc, f"declared but absent from default.json: {only_dc}"
    assert not only_js, f"in default.json but not a dataclass field: {only_js}"


@pytest.mark.parametrize("path",
                         [_param(p, KNOWN_DEFAULT_MISMATCHES) for p in sorted(FIELDS)])
def test_dataclass_default_matches_json(path: str) -> None:
    """
    Each dataclass default equals the `configs/default.json` value.

    Property: the JSON wins at runtime, so a stale dataclass default silently
    misleads anyone reading the source, and becomes load-bearing the moment a
    config is built from a partial dict (`config_from_dict`), where absent keys
    fall back to it. Source: verification plan Stage 1, task 1.
    """
    dc_default = _field_default(FIELDS[path])
    json_value = JSON_VALUES[path]
    if isinstance(dc_default, (int, float)) and not isinstance(dc_default, bool):
        assert float(dc_default) == float(json_value), (
            f"{path}: dataclass={dc_default!r} json={json_value!r}")
    else:
        assert dc_default == json_value, (
            f"{path}: dataclass={dc_default!r} json={json_value!r}")


def test_known_default_mismatches_are_still_real() -> None:
    """
    Every `KNOWN_DEFAULT_MISMATCHES` entry still describes reality.

    Stops the registry from going stale in the other direction: an entry whose
    recorded pair no longer matches means the drift changed shape and the
    xfail above is masking something new.
    """
    wrong = []
    for path, (want_dc, want_js) in KNOWN_DEFAULT_MISMATCHES.items():
        if path not in FIELDS:
            wrong.append(f"{path}: no such field")
            continue
        got_dc, got_js = _field_default(FIELDS[path]), JSON_VALUES.get(path)
        if (got_dc, got_js) != (want_dc, want_js):
            wrong.append(f"{path}: recorded {(want_dc, want_js)}, found {(got_dc, got_js)}")
    assert not wrong, "stale KNOWN_DEFAULT_MISMATCHES entries:\n  " + "\n  ".join(wrong)


# =============================================================================
# 2. PARAM_REGISTRY
# =============================================================================

@pytest.mark.parametrize("path",
                         [_param(p, KNOWN_UNDOCUMENTED) for p in sorted(FIELDS)])
def test_field_is_documented(path: str) -> None:
    """
    Every config field has a `PARAM_REGISTRY` entry.

    `PARAM_REGISTRY` is the single source for `docs/config_reference.md`, so an
    undocumented field is invisible to anyone reading the docs.
    """
    assert path in PARAM_REGISTRY, f"{path} missing from PARAM_REGISTRY"


def test_no_phantom_registry_entries() -> None:
    """Every `PARAM_REGISTRY` key corresponds to a real dataclass field."""
    phantom = sorted(set(PARAM_REGISTRY) - set(FIELDS))
    assert not phantom, f"PARAM_REGISTRY documents non-existent fields: {phantom}"


@pytest.mark.parametrize("path", sorted(PARAM_REGISTRY))
def test_registry_entry_is_complete(path: str) -> None:
    """Each registry entry carries a non-empty help, unit, and range."""
    entry = PARAM_REGISTRY[path]
    for key in ("help", "unit", "range"):
        assert key in entry and str(entry[key]).strip(), f"{path}: empty {key!r}"


# =============================================================================
# 3. Loader semantics
# =============================================================================

def test_load_default_config_matches_json_exactly(default_cfg) -> None:
    """`load_config` reproduces every value in `configs/default.json`."""
    loaded = _walk_json(default_cfg.to_dict())
    assert loaded == JSON_VALUES


def test_to_dict_round_trip(default_cfg) -> None:
    """`to_dict → config_from_dict → to_dict` is the identity."""
    d1 = default_cfg.to_dict()
    assert config_from_dict(d1).to_dict() == d1


def test_deep_merge_does_not_mutate_base() -> None:
    """`_deep_merge` leaves its `base` argument untouched."""
    base = {"a": {"x": 1}, "b": 2}
    snapshot = json.dumps(base, sort_keys=True)
    _deep_merge(base, {"a": {"y": 3}, "b": 9})
    assert json.dumps(base, sort_keys=True) == snapshot


def test_deep_merge_is_recursive_and_override_wins() -> None:
    """Nested dicts merge key-wise; scalars are replaced by the override."""
    merged = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 99}})
    assert merged == {"a": {"x": 1, "y": 99}}


def test_experiment_override_only_touches_named_fields() -> None:
    """
    A partial experiment config changes exactly the keys it names.

    This is the contract every `configs/exp_*.json` and
    `configs/scenarios/stress.json` relies on.
    """
    base = json.loads(DEFAULT_JSON.read_text())
    cfg = config_from_dict(_deep_merge(base, {"topology": {"n_ue": 7}}))
    after = _walk_json(cfg.to_dict())
    changed = {k for k in after if after[k] != JSON_VALUES[k]}
    assert changed == {"topology.n_ue"}
    assert cfg.topology.n_ue == 7


def test_optional_field_accepts_none_and_a_value() -> None:
    """`sensing.sigma_clt` is `Optional[float]`: null → None, 0.1 → 0.1."""
    base = json.loads(DEFAULT_JSON.read_text())
    assert config_from_dict(base).sensing.sigma_clt is None
    cfg = config_from_dict(_deep_merge(base, {"sensing": {"sigma_clt": 0.1}}))
    assert cfg.sensing.sigma_clt == pytest.approx(0.1)


@pytest.mark.xfail(strict=True,
                   reason="F-00-06: unknown keys are silently dropped by "
                          "_populate_dataclass, so a typo'd override is inert")
def test_unknown_json_key_is_rejected() -> None:
    """
    A key that is not a config field should be an error, not a silent no-op.

    Today `_populate_dataclass` ignores it ("forward compatibility"), so
    `{"topology": {"n_aps": 3}}` leaves `n_ap` at its default and nothing
    anywhere says so.
    """
    base = json.loads(DEFAULT_JSON.read_text())
    with pytest.raises((ValueError, TypeError, KeyError)):
        config_from_dict(_deep_merge(base, {"topology": {"n_aps": 3}}))


def test_unknown_json_key_is_currently_silent() -> None:
    """
    Pins the *actual* behavior behind F-00-06 so the blast radius is explicit.

    Delete this test when `test_unknown_json_key_is_rejected` starts passing.
    """
    base = json.loads(DEFAULT_JSON.read_text())
    cfg = config_from_dict(_deep_merge(base, {"topology": {"n_aps": 3}}))
    assert cfg.topology.n_ap == JSON_VALUES["topology.n_ap"]
    assert not hasattr(cfg.topology, "n_aps")


@pytest.mark.xfail(strict=True,
                   reason="F-00-06: config dataclasses are not frozen and have "
                          "no __slots__, so any attribute name is accepted")
def test_unknown_attribute_assignment_is_rejected(tiny_cfg) -> None:
    """Setting a field that does not exist should raise, not create it."""
    with pytest.raises((AttributeError, TypeError)):
        tiny_cfg.algorithm.split.gamma_db = 10.0   # the F-00-05 typo, verbatim


@pytest.mark.xfail(strict=True,
                   reason="F-01-03: no type coercion, an int field silently "
                          "accepts a float")
def test_int_field_rejects_float() -> None:
    """`topology.n_ap = 7.5` should fail; today it yields a float `n_ap`."""
    base = json.loads(DEFAULT_JSON.read_text())
    with pytest.raises((ValueError, TypeError)):
        config_from_dict(_deep_merge(base, {"topology": {"n_ap": 7.5}}))


@pytest.mark.xfail(strict=True,
                   reason="F-01-03: a non-Optional field silently accepts null")
def test_non_optional_field_rejects_null() -> None:
    """`channel.snr_db = null` should fail; today it yields `None`."""
    base = json.loads(DEFAULT_JSON.read_text())
    with pytest.raises((ValueError, TypeError)):
        config_from_dict(_deep_merge(base, {"channel": {"snr_db": None}}))


# =============================================================================
# 4. validate()
# =============================================================================

@pytest.mark.parametrize("label,mutate", [
    ("n_rf_chains > n_ant",  lambda c: setattr(c.topology, "n_rf_chains", 999)),
    ("n_sensing_rx >= n_ap", lambda c: setattr(c.topology, "n_sensing_rx", 99)),
    ("bad array_type",       lambda c: setattr(c.topology, "array_type", "HEX")),
    ("negative admm kappa",  lambda c: setattr(c.algorithm.admm, "kappa", -1.0)),
    ("negative split kappa", lambda c: setattr(c.algorithm.split, "kappa", -1.0)),
    ("bad estimation_method",
     lambda c: setattr(c.channel, "estimation_method", "XYZ")),
    ("n_trials < 1",         lambda c: setattr(c.simulation, "n_trials", 0)),
    ("negative ue separation",
     lambda c: setattr(c.topology, "ue_min_separation_m", -1.0)),
])
def test_validate_rejects(default_cfg, label: str, mutate) -> None:
    """`CORDISConfig.validate()` rejects each documented invalid combination."""
    import copy
    cfg = copy.deepcopy(default_cfg)
    mutate(cfg)
    with pytest.raises(ValueError):
        cfg.validate()


def test_default_config_validates(default_cfg) -> None:
    """The shipped default config passes its own validation."""
    default_cfg.validate()


# =============================================================================
# 5. Shipped config files
# =============================================================================

CONFIG_FILES = sorted(
    [p for p in (REPO_ROOT / "configs").glob("*.json")]
    + [p for p in (REPO_ROOT / "configs" / "scenarios").glob("*.json")]
)


@pytest.mark.parametrize("path", CONFIG_FILES,
                         ids=[p.relative_to(REPO_ROOT).as_posix() for p in CONFIG_FILES])
def test_shipped_config_loads_and_has_no_phantom_keys(path: Path) -> None:
    """
    Every committed config parses, merges onto the default, and validates,
    and contains no key that the loader would silently drop.
    """
    keys = _walk_json(json.loads(path.read_text()))
    phantom = sorted(set(keys) - set(FIELDS))
    assert not phantom, f"{path.name} has keys that are not config fields: {phantom}"
    if path.name != "default.json":
        load_config("configs/default.json", path.relative_to(REPO_ROOT).as_posix())


def test_stress_scenario_is_the_documented_harder_regime() -> None:
    """
    `configs/scenarios/stress.json` encodes the disclosure run described in
    CLAUDE.md: fewer/smaller APs, weaker pilots, clutter on the target, κ=1.
    """
    cfg = load_config("configs/default.json", "configs/scenarios/stress.json")
    assert cfg.topology.n_ap == 6
    assert cfg.topology.n_ant == 10
    assert cfg.channel.pilot_power_db == pytest.approx(120.0)
    assert cfg.sensing.clutter_center_strategy == "target_centroid"
    assert cfg.algorithm.admm.kappa == pytest.approx(1.0)


# =============================================================================
# 6. configs/recipes/_defaults.sh: the shell mirror
# =============================================================================

def _recipe_var_defaults() -> dict[str, str]:
    txt = RECIPE_DEFAULTS.read_text()
    return dict(re.findall(r'^([A-Z_][A-Z0-9_]*)="\$\{\1:-(.*?)\}"', txt, re.M))


def _recipe_set_args() -> dict[str, str]:
    txt = RECIPE_DEFAULTS.read_text()
    block = txt.split("SET_ARGS=(")[1].split("\n)")[0]
    return dict(re.findall(r'^\s*([\w.]+)="\$([A-Z_][A-Z0-9_]*)"', block, re.M))


def _norm(x: Any) -> Any:
    s = str(x).strip().lower()
    if s in ("null", "none"):
        return None
    if s in ("true", "false"):
        return s == "true"
    try:
        return float(s)
    except ValueError:
        return s


SET_ARGS = _recipe_set_args()


def test_recipe_set_args_were_parsed() -> None:
    """Guard the regex: if `_defaults.sh` is reformatted, fail loudly here."""
    assert len(SET_ARGS) > 60, f"only parsed {len(SET_ARGS)} SET_ARGS entries"


@pytest.mark.parametrize("path", sorted(SET_ARGS), ids=sorted(SET_ARGS))
def test_recipe_set_arg_targets_a_real_field(path: str) -> None:
    """
    Every `--set <path>=` in `_defaults.sh` names a real config field.

    A stale dotted path here is invisible: `create_config.py` only warns, and
    the loader then drops the key (F-00-06). This is the check that would have
    caught the `algorithm.split.gamma_db` class of bug.
    """
    assert path in FIELDS, f"SET_ARGS targets a non-existent field: {path}"


@pytest.mark.parametrize("path", sorted(SET_ARGS), ids=sorted(SET_ARGS))
def test_recipe_default_matches_json(path: str) -> None:
    """
    Each recipe fallback `${VAR:-x}` equals the `configs/default.json` value.

    `_defaults.sh` is a hand-maintained mirror; drift here silently changes
    every generated experiment config.
    """
    var = SET_ARGS[path]
    recipe_value = _recipe_var_defaults().get(var)
    assert recipe_value is not None, f"{var} has no ${{VAR:-default}} definition"
    assert _norm(recipe_value) == _norm(JSON_VALUES[path]), (
        f"{path}: recipe ${var}={recipe_value!r} vs default.json "
        f"{JSON_VALUES[path]!r}")


def test_every_recipe_var_is_used() -> None:
    """No `${VAR:-default}` is defined and then never referenced in SET_ARGS."""
    defined = set(_recipe_var_defaults()) - {"NAME", "DRY_RUN_FLAG"}
    unused = sorted(defined - set(SET_ARGS.values()))
    assert not unused, f"defined but never used in SET_ARGS: {unused}"


# =============================================================================
# 7. Sentinel-default policy
# =============================================================================

def test_admm_spec_emits_no_tuning_knobs_by_default() -> None:
    """
    `admm_spec()` with no explicit knob adds none to `spec.params`.

    This is the sentinel pattern that replaced the `DEFAULT_*` constants: with
    no params, the algorithm reads `cfg.algorithm.admm.*`, keeping the JSON the
    single source of truth.
    """
    from cordis.experiments.specs import admm_spec
    spec = admm_spec(n_ue=4)
    knobs = {"kappa", "rho_admm", "n_admm_max", "eps_pri", "eps_dual", "xi_slack"}
    assert not (knobs & set(spec.params)), f"leaked: {knobs & set(spec.params)}"


def test_admm_spec_forwards_explicit_knobs() -> None:
    """An explicitly passed knob does appear in `spec.params`."""
    from cordis.experiments.specs import admm_spec
    assert admm_spec(n_ue=4, kappa=0.5).params["kappa"] == pytest.approx(0.5)


def test_admm_kwargs_from_cfg_mirrors_the_config(default_cfg) -> None:
    """`_admm_kwargs_from_cfg` forwards the config's own ADMM values."""
    from cordis.experiments.registry import _admm_kwargs_from_cfg
    kw = _admm_kwargs_from_cfg(default_cfg)
    a = default_cfg.algorithm.admm
    assert kw["kappa"] == pytest.approx(a.kappa)
    assert kw["rho_admm"] == pytest.approx(a.rho)
    assert kw["n_admm_max"] == a.n_max
    assert kw["eps_pri"] == pytest.approx(a.eps_pri)
    assert kw["eps_dual"] == pytest.approx(a.eps_dual)
    assert kw["xi_slack"] == pytest.approx(a.xi_slack)


#: `file:line -> (field, fallback, configs/default.json value)` for every
#: three-argument `getattr(cfg…, field, fallback)` whose fallback contradicts
#: the config. F-01-04. Keyed by field so a line-number shift does not break it.
KNOWN_SHADOW_DEFAULTS: dict[str, tuple[Any, Any]] = {
    "algorithm.admm.early_stop_patience":   (15, 30),
    "algorithm.admm.best_iter_criterion":   ("residual_norm", "feasible_then_residual"),
    "algorithm.admm.adaptive_rho":          (True, False),
    "sensing.clutter_center_strategy":      ("target_centroid", "offset"),
    "topology.n_ant":                       (0, 16),
    "topology.n_ue":                        (0, 4),
    "topology.n_targets":                   (0, 2),
    "simulation.n_trials":                  (None, 100),
}


def _scan_shadow_defaults() -> list[tuple[str, str, Any]]:
    """(location, field name, fallback) for every 3-arg getattr on a config."""
    import ast
    found: list[tuple[str, str, Any]] = []
    roots = [REPO_ROOT / "cordis", REPO_ROOT / "scripts"]
    for root in roots:
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) == 3):
                    continue
                target = ast.unparse(node.args[0])
                if "cfg" not in target and "config" not in target:
                    continue
                try:
                    name = ast.literal_eval(node.args[1])
                    fallback = ast.literal_eval(node.args[2])
                except Exception:
                    continue
                found.append(
                    (f"{p.relative_to(REPO_ROOT)}:{node.lineno}", name, fallback))
    return found


LEAF_TO_PATH: dict[str, str] = {}
for _p in FIELDS:
    LEAF_TO_PATH.setdefault(_p.rsplit(".", 1)[-1], _p)


@pytest.mark.xfail(strict=True,
                   reason="F-01-04: getattr(cfg, field, fallback) fallbacks that "
                          "contradict configs/default.json")
def test_no_getattr_fallback_contradicts_the_config() -> None:
    """
    A defensive `getattr(cfg.x, "field", <fallback>)` must not encode a
    *different* value than the config declares.

    Three of the current offenders would silently flip settled, evidenced
    choices: `adaptive_rho → True`, `best_iter_criterion → "residual_norm"`,
    and `clutter_center_strategy → "target_centroid"` (the last collapses the
    κ gain that Fig. 4 reports). They are unreachable for a real
    `CORDISConfig`, every declared field always exists, but they are the
    only written record of a default in those files, and they are wrong.
    """
    bad = []
    for loc, name, fallback in _scan_shadow_defaults():
        path = LEAF_TO_PATH.get(name)
        if path is None:
            continue
        expected = JSON_VALUES[path]
        if fallback != expected:
            bad.append(f"{loc}: getattr(..., {name!r}, {fallback!r}) "
                       f"but default.json says {expected!r}")
    assert not bad, "shadow defaults:\n  " + "\n  ".join(bad)


def test_known_shadow_defaults_are_still_real() -> None:
    """
    Every `KNOWN_SHADOW_DEFAULTS` entry still describes a live offender.

    Keeps the xfail above honest: if one is fixed, this fails and points at the
    entry to delete.
    """
    live = {LEAF_TO_PATH.get(name): fb
            for _, name, fb in _scan_shadow_defaults()
            if LEAF_TO_PATH.get(name)}
    stale = []
    for path, (fallback, json_value) in KNOWN_SHADOW_DEFAULTS.items():
        if path not in live:
            stale.append(f"{path}: no longer has a 3-arg getattr fallback")
        elif live[path] != fallback:
            stale.append(f"{path}: recorded fallback {fallback!r}, found {live[path]!r}")
        elif JSON_VALUES.get(path) != json_value:
            stale.append(f"{path}: recorded json {json_value!r}, "
                         f"found {JSON_VALUES.get(path)!r}")
    assert not stale, "stale KNOWN_SHADOW_DEFAULTS entries:\n  " + "\n  ".join(stale)


def test_no_default_constants_in_package() -> None:
    """
    No module-level `DEFAULT_*` constant exists anywhere in `cordis/`.

    Reintroducing one recreates the out-of-sync-defaults bug the sentinel
    pattern was written to kill (CLAUDE.md, "Config system").
    """
    offenders = []
    for p in sorted((REPO_ROOT / "cordis").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"^\s*(DEFAULT_[A-Z_0-9]+)\s*[:=]", line):
                offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, "DEFAULT_* constants found:\n  " + "\n  ".join(offenders)
