"""Live ISI-violation columns: same formula as HM_Tracker step r, computed from
the CURRENT spikes of a cluster, so a merged or split cluster shows its own
value instead of the one copied from its parent's .tsv row."""

import numpy as np

from ..isi_metrics import drop_duplicated_spikes, isi_violations
from .test_base import MyController, MyModel


def test_drop_duplicated_spikes():
    # 0.3 ms at 30 kHz = 9 samples. 105 and 108 follow 100 too closely, 209
    # follows 200 by exactly 9 samples; the first spike of each run is kept.
    samples = [219, 100, 105, 108, 200, 209]
    assert drop_duplicated_spikes(samples, 30000).tolist() == [100, 200, 219]


def test_isi_violations_formula():
    fs, duration = 30000.0, 100.0
    # The spike at sample 3 is a duplicate of 0 and is dropped first. Of the 4
    # spikes left, one ISI is 1 ms (30 samples) -> 1 violation.
    samples = [0, 3, 30, 300000, 600000]
    ratio, count, n_spikes = isi_violations(samples, fs, duration)
    assert (count, n_spikes) == (1, 4)
    # spikeinterface: violation_rate / total_rate
    expected = (1 / (2 * 4 * 0.0015)) / (4 / duration)
    assert np.isclose(ratio, expected)


def test_isi_violations_empty_cluster():
    ratio, count, n_spikes = isi_violations([], 30000.0, 10.0)
    assert np.isnan(ratio)
    assert (count, n_spikes) == (0, 0)


def _expected(controller, cluster_id):
    model = controller.model
    spike_ids = controller.supervisor.clustering.spikes_per_cluster[cluster_id]
    return isi_violations(model.spike_samples[spike_ids], model.sample_rate, model.duration)


def test_live_columns_replace_stale_tsv_values(tempdir):
    model = MyModel()
    # Own copy: a merge edits spike_clusters in place, and MyModel's arrays are
    # shared by every test.
    model.spike_clusters = model.spike_clusters.copy()
    cluster_ids = np.unique(model.spike_clusters)
    # What step r's cluster_<metric>.tsv files would hold (99 = stale value).
    model.metadata = {
        'group': {3: 'noise'},
        'isi_violations_ratio': {int(c): 99.0 for c in cluster_ids},
        'snr': {int(c): 5.0 for c in cluster_ids},
    }
    controller = MyController(
        dir_path=tempdir, config_dir=tempdir / 'config', model=model,
        clear_cache=True, enable_threading=False)
    supervisor = controller.supervisor

    # The live value is shown, once; the stale .tsv value is not loaded at all
    # (so it is never copied on merge/split, nor saved back).
    first = int(cluster_ids[0])
    info = supervisor.get_cluster_info(first)
    ratio, count, n_spikes = _expected(controller, first)
    assert info['isi_violations_ratio'] == ratio != 99.0
    assert info['isi_violations_count'] == count
    assert info['num_spikes'] == n_spikes
    assert supervisor.columns.count('isi_violations_ratio') == 1
    assert supervisor.columns[:2] == ['id', 'isi_violations_ratio']   # first after id
    assert 'isi_violations_ratio' not in supervisor.cluster_meta.fields
    assert 'snr' in supervisor.cluster_meta.fields    # other .tsv columns still load

    # A merged cluster gets the value of its own spikes right away.
    up = supervisor.merge([int(cluster_ids[0]), int(cluster_ids[1])])
    merged = up.added[0]
    assert controller.get_isi_violations_ratio(merged) == _expected(controller, merged)[0]
    assert controller._get_isi(merged).text.startswith('ratio ')

    controller.close()
