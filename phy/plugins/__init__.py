"""Plugins bundled with phy.

Importing this package imports every bundled plugin module, which is what
registers the plugin classes in `IPluginRegistry`. `attach_plugins()` imports it
before resolving plugin names, so a bundled plugin can be enabled by name -- via
the `--recluster` flag or the user config -- without configuring `Plugins.dirs`.

Bundled plugin modules must stay cheap to import: heavy optional dependencies
(scikit-learn, isosplit, ...) are imported inside functions, never at module
level, so that `import phy` does not pull them in.
"""

from . import (
    recluster,  # noqa: F401
    sort_by_id,  # noqa: F401
)
