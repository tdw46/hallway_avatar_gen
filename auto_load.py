"""
Reusable Blender add-on auto loader and hot reloader for large, multi-file add-ons.

Usage (in your package's __init__.py):

    from . import auto_load

    def register():
        # Auto-discover submodules, compute dependency order, reload, and
        # register all Blender classes.
        auto_load.register()

        # Add package-specific Scene properties AFTER class registration.
        props_mod = auto_load.get_module("properties")
        if props_mod and hasattr(props_mod, "register_properties"):
            props_mod.register_properties()

    def unregister():
        # Optional: cancel long-running tasks/state BEFORE removing classes.
        utils_mod = auto_load.get_module("utils")
        try:
            if utils_mod and hasattr(utils_mod, "batch_state") and utils_mod.batch_state.is_processing:
                utils_mod.batch_state.should_cancel = True
                utils_mod.batch_state.is_processing = False
        except Exception:
            pass

        # Remove Scene properties BEFORE class unregistration.
        props_mod = auto_load.get_module("properties")
        if props_mod and hasattr(props_mod, "unregister_properties"):
            try:
                props_mod.unregister_properties()
            except Exception as e:
                print(f"YourAddon: failed to unregister properties: {e}")

        # Finally unregister all classes.
        auto_load.unregister()

        # Clean up sys.modules to ensure fresh reloads on next enable/import
        # Purge this package from sys.modules so next enable/import loads fresh code for all submodules
        try:
            pkg = __package__
            to_del = [name for name in list(sys.modules.keys()) if name == pkg or name.startswith(pkg + ".")]
            for name in to_del:
                del sys.modules[name]
            importlib.invalidate_caches()
            # Note: this is safe & targeted; avoids global sys.modules purges.
        except Exception as e:
            print(f"AutoLoad: sys.modules purge warning: {e}")

Key features
- Auto-discovers all submodules in the package (recursively).
- Builds an intra-package import graph (AST) and reloads modules in a dependency-safe order.
- Discovers Blender classes (PropertyGroup, AddonPreferences, UIList, GizmoGroup, Operator, Menu, Panel).
- Registers PropertyGroup classes using a dependency-aware topological sort, with progressive retries.
- Registers remaining classes in a stable priority order.
- Cleanly unregisters in reverse order.

Advanced
- get_module(name): access any loaded submodule by basename (supports nested modules).
- set_excludes([...]): exclude module basenames from auto-discovery (default: "__init__", "auto_load", "__pycache__").
- set_modules([...]): optionally override the discovery/graph order with an explicit list, if needed.

Notes
- Blenderâ€™s built-in refresh operators donâ€™t reload active add-on code; the standard pattern is refresh â†’ disable/enable.
- To guarantee fresh code without restarting Blender, purge your package from sys.modules and call
  importlib.invalidate_caches() in your package's unregister() (see this add-onâ€™s __init__.py).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import ast
from typing import Dict, Iterable, List, Sequence, Tuple

import bpy

# Internal state
_ORDERED_MODULES: List[str] = []
_MODULES: Dict[str, object] = {}
_REGISTERED_CLASSES: List[type] = []
_EXCLUDES: Tuple[str, ...] = ("__init__", "auto_load", "__pycache__")


def set_modules(module_names: Sequence[str]) -> None:
    """Set the explicit import/reload order for submodules.

    Pass a list of basenames (without package prefix), e.g. ["utils", "properties", ...].
    """
    global _ORDERED_MODULES
    _ORDERED_MODULES = list(module_names)


def set_excludes(excludes: Sequence[str]) -> None:
    """Set module basenames to exclude from auto-discovery."""
    global _EXCLUDES
    _EXCLUDES = tuple(excludes)


def _package_name() -> str:
    # This module lives inside the add-on package, so __package__ is the package name.
    pkg = __package__
    if not pkg:
        # Fallback: derive from module name
        pkg = __name__.rpartition(".")[0]
    return pkg


def _reload_or_import_relative(mod_name: str, package: str):
    """Import a submodule relative to our package and reload if already present.

    Using relative import prevents issues with non-identifier package names.
    """
    # Always import relative to the package containing this auto_load.py
    module = importlib.import_module(f".{mod_name}", package=package)
    # Ensure the latest code is active by reloading the module object we just imported
    module = importlib.reload(module)
    return module


def _import_modules() -> None:
    """Import or reload submodules.

    If explicit modules were set via set_modules(), they are used.
    Otherwise, auto-discover modules and compute a dependency-based reload order.
    """
    importlib.invalidate_caches()
    pkg = _package_name()

    if _ORDERED_MODULES:
        for mod_name in _ORDERED_MODULES:
            module = _reload_or_import_relative(mod_name, pkg)
            _MODULES[mod_name] = module
        return

    # Auto-discovery path
    discovered = _discover_module_names()

    # Import all discovered modules (initial import)
    for name in discovered:
        try:
            module = importlib.import_module(f".{name}", package=pkg)
            _MODULES[name] = module
        except Exception as e:
            print(f"Auto_load: initial import failed for {name}: {e}")

    # Build dependency graph and compute reload order
    dep_map = _module_dep_graph(discovered)
    reload_order = _toposort_modules(discovered, dep_map)

    # Now reload modules in order to ensure providers come first
    for name in reload_order:
        try:
            module = importlib.reload(_MODULES[name])
            _MODULES[name] = module
        except Exception as e:
            print(f"Auto_load: reload failed for {name}: {e}")


def _discover_module_names() -> List[str]:
    """Discover all submodules under this package (recursively).

    Returns module names relative to the package, e.g., 'utils', 'ops_image_gen', 'subpkg.mod'.
    Excludes '__init__', 'auto_load', '__pycache__', and names starting with '_'.
    """
    pkg_name = _package_name()
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:
        return []

    result: List[str] = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        full_name = mod.name  # e.g., 'package.ops_image_gen'
        rel = full_name[len(pkg_name) + 1 :]  # strip 'package.'
        base = rel.rsplit(".", 1)[-1]
        if base in _EXCLUDES:
            continue
        if base.startswith("_"):
            continue
        result.append(rel)

    # Prefer flat top-level modules before nested ones to reduce surprises
    result.sort(key=lambda n: (n.count("."), n))
    return result


def _resolve_relative_import(current: str, level: int, module: str | None) -> str | None:
    """Resolve a relative import to a package-relative dotted name.

    current: 'a.b.c' (relative to package), level: 1 means 'from .', 2 means 'from ..', etc.
    module: the module path in the import-from statement (may be None).
    """
    parts = current.split(".")
    if level > len(parts):
        return None
    base_parts = parts[: len(parts) - level]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts) if base_parts else None


def _module_dep_graph(module_names: List[str]) -> Dict[str, List[str]]:
    """Build a dependency map among discovered modules using AST import analysis."""
    pkg_name = _package_name()
    name_set = set(module_names)
    dep_map: Dict[str, List[str]] = {name: [] for name in module_names}

    for name in module_names:
        mod = _MODULES.get(name)
        if not mod:
            continue
        try:
            source = inspect.getsource(mod)
        except Exception:
            # Fallback: try loading source from spec.origin
            try:
                spec = importlib.util.find_spec(f"{pkg_name}.{name}")
                if not spec or not spec.origin:
                    continue
                with open(spec.origin, "r", encoding="utf-8") as f:
                    source = f.read()
            except Exception:
                continue

        try:
            tree = ast.parse(source)
        except Exception:
            continue

        current = name  # relative to package
        deps: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Handle 'from .x import ...' or 'from ..x import ...'
                if node.level and node.level > 0:
                    rel = _resolve_relative_import(current, node.level, node.module)
                    if rel:
                        # If importing from a submodule (e.g., 'a.b'), depend on that submodule root
                        dep = rel
                        if dep in name_set and dep not in deps:
                            deps.append(dep)
                else:
                    # Absolute import
                    if node.module and node.module.startswith(pkg_name + "."):
                        dep = node.module[len(pkg_name) + 1 :]
                        if dep in name_set and dep not in deps:
                            deps.append(dep)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    fullname = alias.name
                    if fullname.startswith(pkg_name + "."):
                        dep = fullname[len(pkg_name) + 1 :]
                        if dep in name_set and dep not in deps:
                            deps.append(dep)

        dep_map[name] = deps

    return dep_map


def _toposort_modules(names: List[str], dep_map: Dict[str, List[str]]) -> List[str]:
    """Topologically sort modules by their intra-package import dependencies.

    Stable: preserves original discovery order among independent nodes.
    """
    index = {n: i for i, n in enumerate(names)}
    indeg = {n: 0 for n in names}
    for n in names:
        for d in dep_map.get(n, []):
            indeg[n] += 1

    ready = [n for n in names if indeg[n] == 0]
    ready.sort(key=lambda n: index[n])
    result: List[str] = []

    while ready:
        n = ready.pop(0)
        result.append(n)
        for m in names:
            if n in dep_map.get(m, []):
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
                    ready.sort(key=lambda k: index[k])

    if len(result) != len(names):
        # Cycle detected; fall back to discovered order
        return list(names)
    return result


def _iter_registerable_classes() -> Iterable[type]:
    """Yield registerable RNA classes discovered across the loaded submodules."""
    from bpy.types import AddonPreferences, Menu, Operator, Panel, PropertyGroup, UIList

    # Optional types (not available in all contexts)
    try:
        from bpy.types import GizmoGroup  # type: ignore
    except Exception:  # pragma: no cover - optional in some Blender versions
        GizmoGroup = None  # type: ignore

    base_types: Tuple[type, ...] = (
        PropertyGroup,
        AddonPreferences,
        UIList,
        Operator,
        Menu,
        Panel,
    )
    if GizmoGroup:
        base_types = base_types + (GizmoGroup,)  # type: ignore

    for module in _MODULES.values():
        # Only consider classes defined in the module itself
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                try:
                    if any(issubclass(obj, bt) for bt in base_types):
                        yield obj
                except Exception:
                    # Defensive: issubclass may fail on certain Blender RNA proxy types
                    continue


def _class_priority(cls: type) -> int:
    """Return a sort key so dependent types register first.

    Lower numbers register earlier.
    """
    from bpy.types import AddonPreferences, Menu, Operator, Panel, PropertyGroup, UIList

    try:
        from bpy.types import GizmoGroup  # type: ignore
    except Exception:
        GizmoGroup = None  # type: ignore

    try:
        if issubclass(cls, PropertyGroup):
            return 10
        if issubclass(cls, AddonPreferences):
            return 20
        if issubclass(cls, UIList):
            return 30
        if GizmoGroup and issubclass(cls, GizmoGroup):  # type: ignore
            return 35
        if issubclass(cls, Operator):
            return 40
        if issubclass(cls, Menu):
            return 45
        if issubclass(cls, Panel):
            # Ensure parent panels (no bl_parent_id) register before child subpanels.
            # Blender requires the parent panel's bl_idname to already be registered
            # when registering a child that references it via bl_parent_id.
            if getattr(cls, "bl_parent_id", None):
                # Child / subpanels come slightly later than their parents.
                return 55
            # Top-level panels first.
            return 50
    except Exception:
        pass
    return 100


def _propertygroup_dependencies(pg_cls: type) -> List[type]:
    """Return a list of PropertyGroup classes that this PropertyGroup depends on.

    Detects dependencies via CollectionProperty(type=OtherPG) and PointerProperty(type=OtherPG)
    by inspecting the class annotations (bpy uses annotations to declare properties).
    """
    from bpy.types import PropertyGroup

    deps: List[type] = []
    ann = getattr(pg_cls, "__annotations__", {}) or {}

    for prop_val in ann.values():
        try:
            # Blender stores property declarations as deferred property objects
            # which expose keyword args via `.keywords`.
            dep_type = None
            kw = getattr(prop_val, "keywords", None)
            if kw:
                dep_type = kw.get("type")
            if dep_type is None:
                # Some Blender versions expose fixed_type or type attributes
                dep_type = getattr(prop_val, "fixed_type", None) or getattr(prop_val, "type", None)
            if isinstance(dep_type, type) and issubclass(dep_type, PropertyGroup):
                deps.append(dep_type)
        except Exception:
            continue

    return deps


def _toposort_propertygroups(pgs: List[type]) -> List[type]:
    """Topologically sort PropertyGroup classes by their PG->PG dependencies.

    If sorting fails (cycles/unexpected), fall back to name sort.
    """
    # Build adjacency among the provided set only
    set_pgs = set(pgs)
    dep_map = {pg: [d for d in _propertygroup_dependencies(pg) if d in set_pgs] for pg in pgs}

    # Kahn's algorithm
    indeg = {pg: 0 for pg in pgs}
    for pg, deps in dep_map.items():
        for d in deps:
            indeg[pg] += 1

    # Start with zero indegree nodes; preserve original declaration order for determinism
    orig_index = {pg: i for i, pg in enumerate(pgs)}
    ready = [pg for pg, deg in indeg.items() if deg == 0]
    ready.sort(key=lambda c: orig_index.get(c, 0))
    result: List[type] = []

    while ready:
        n = ready.pop(0)
        result.append(n)
        # remove edges n -> others (where others depend on n)
        for m, deps in dep_map.items():
            if n in deps:
                indeg[m] -= 1
                if indeg[m] == 0:
                    # keep deterministic order based on original order
                    ready.append(m)
                    ready.sort(key=lambda c: orig_index.get(c, 0))

    if len(result) != len(pgs):
        # Cycle or unexpected state; fallback to original declaration order
        return list(pgs)

    return result


def _sorted_classes(classes: List[type]) -> List[type]:
    """Return classes sorted by priority with PropertyGroups topologically sorted."""
    from bpy.types import PropertyGroup

    pgs: List[type] = []
    others: List[type] = []
    for c in classes:
        try:
            if issubclass(c, PropertyGroup):
                pgs.append(c)
            else:
                others.append(c)
        except Exception:
            others.append(c)

    # Baseline order for PGs: by module order, then source line, then name
    mod_index = {name: i for i, name in enumerate(_ORDERED_MODULES)}

    def _pg_order_key(cls: type) -> Tuple[int, int, str]:
        mod = getattr(cls, "__module__", "")
        base = mod.rsplit(".", 1)[-1]
        idx = mod_index.get(base, 10_000)
        try:
            _, line = inspect.getsourcelines(cls)
        except Exception:
            line = 0
        return (idx, line, cls.__name__)

    pgs.sort(key=_pg_order_key)
    pgs_sorted = _toposort_propertygroups(pgs)
    others_sorted = sorted(others, key=lambda c: (_class_priority(c), c.__name__))
    return pgs_sorted + others_sorted


def register() -> None:
    """Import/reload submodules, discover classes and register them."""
    _import_modules()

    all_classes = list(_iter_registerable_classes())

    # Split PGs and others; PGs will be registered first using progressive passes
    from bpy.types import PropertyGroup
    pgs = []
    others = []
    for c in all_classes:
        try:
            if issubclass(c, PropertyGroup):
                pgs.append(c)
            else:
                others.append(c)
        except Exception:
            others.append(c)

    # Determine a stable plan for PGs (try topo sort, but registration will also be progressive)
    pgs_plan = _toposort_propertygroups(pgs)

    # Debug: show registration plan (especially PropertyGroups order)
    try:
        print(f"Auto_load: registering {len(all_classes)} classes. PropertyGroups order: {[c.__name__ for c in pgs_plan]}")
    except Exception:
        print(f"Auto_load: registering {len(all_classes)} classes")

    registered: List[type] = []

    # Progressive registration for PropertyGroups to satisfy dependencies
    remaining = list(pgs_plan)
    max_passes = len(remaining) + 5
    passes = 0
    while remaining and passes < max_passes:
        passes += 1
        next_remaining: List[type] = []
        for cls in remaining:
            try:
                bpy.utils.register_class(cls)
                registered.append(cls)
            except Exception:
                # Defer this class to a later pass
                next_remaining.append(cls)
        if len(next_remaining) == len(remaining):
            # No progress; break to avoid infinite loop
            break
        remaining = next_remaining

    if remaining:
        # As a final attempt, try to register any still-remaining PGs and let Blender raise
        for cls in remaining:
            bpy.utils.register_class(cls)
            registered.append(cls)

    # Now register remaining classes in priority order
    others_sorted = sorted(others, key=lambda c: (_class_priority(c), c.__name__))
    for cls in others_sorted:
        bpy.utils.register_class(cls)
        registered.append(cls)

    _REGISTERED_CLASSES[:] = registered


def unregister() -> None:
    """Unregister previously registered classes in reverse order."""
    for cls in reversed(_REGISTERED_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            print(f"Auto_load: Failed to unregister {cls.__name__}: {e}")
    _REGISTERED_CLASSES.clear()


def get_module(name: str):
    """Access a loaded submodule by its basename (e.g., get_module('utils')).

    If exact key not found, fall back to matching the last dotted component or suffix.
    """
    mod = _MODULES.get(name)
    if mod is not None:
        return mod
    # Fallbacks for nested names
    for key, module in _MODULES.items():
        if key == name:
            return module
        # last dotted component
        last = key.rsplit(".", 1)[-1]
        if last == name:
            return module
        if key.endswith("." + name):
            return module
    return None


def get_registered_classes() -> Tuple[type, ...]:
    """Return the tuple of registered classes (in registration order)."""
    return tuple(_REGISTERED_CLASSES)
