"""Sort the cluster view by cluster id, so navigation moves by index number.

After a label action -- e.g. `alt+g` (move_best_to_good) -- phy advances to the
next row in the cluster view's *current sort order* (see `_after_move` in
phy/cluster/supervisor.py, which enqueues `cluster_view.next`). When the list is
sorted by the `group` column that means "jump to the next good cluster"; sorting
by `id` instead makes it "jump to the next index number".

This forces the id sort on every launch (it runs after the saved GUI state is
restored, so it wins). That is the point -- index-order navigation regardless of
whatever column was last clicked. Turn it off with `phy template-gui
--no-sort-by-id`, or by not listing this plugin.

Bundled with phy; attached by default for the Template GUI.
"""

import logging

from phy import IPlugin, connect

logger = logging.getLogger('phy')


class SortByIdPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            # 'id' ascending == cluster index order. `supervisor.sort` is the public
            # wrapper around `cluster_view.sort_by`.
            controller.supervisor.sort('id', 'asc')
            logger.debug('Sorted the cluster view by id (ascending).')
