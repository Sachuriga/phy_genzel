"""Live ISI-violation metrics for the cluster view.

These give the same numbers that HM_Tracker step r ("recompute metrics") writes
into cluster_<metric>.tsv. Phy computes them from the cluster's CURRENT spikes,
so after a merge or split the new cluster shows its own value at once. Without
this, phy copies the parent cluster's value from the .tsv file until step r is
run again.

Per cluster:

1. Drop duplicated spikes: a spike at most 0.3 ms after the previous spike of
   the same cluster is the same spike counted twice, and the earlier one is
   kept (spikeinterface ``remove_duplicated_spikes``, method="keep_first").
2. ``isi_violations_count``: the number of ISIs shorter than 1.5 ms.
3. ``isi_violations_ratio`` (Hill et al. 2011, spikeinterface "isi_violation")::

       count * duration / (2 * n_spikes**2 * 1.5 ms)

   This is NOT the fraction of ISIs in the first bin of the ISI view. It
   scales with 1 / firing_rate**2, so a sparse unit gets a high ratio from a
   few violations and a fast unit a low ratio from many.
"""

import numpy as np

DUPLICATE_CENSORED_PERIOD_MS = 0.3
ISI_THRESHOLD_MS = 1.5


def drop_duplicated_spikes(spike_samples, sample_rate,
                           censored_period_ms=DUPLICATE_CENSORED_PERIOD_MS):
    """Sorted spike samples of ONE cluster without its duplicated spikes."""
    samples = np.sort(np.asarray(spike_samples, dtype=np.int64))
    censored_samples = int(round(censored_period_ms * 1e-3 * sample_rate))
    # A spike is a duplicate when it comes too soon after the previous spike
    # (the first spike has no previous one, so it is always kept).
    is_duplicate = np.zeros(len(samples), dtype=bool)
    is_duplicate[1:] = np.diff(samples) <= censored_samples
    return samples[~is_duplicate]


def isi_violations(spike_samples, sample_rate, duration, isi_threshold_ms=ISI_THRESHOLD_MS):
    """Return (ratio, count, n_spikes) for ONE cluster, duplicates dropped.

    `duration` is the whole recording in seconds (the model's duration), as in
    spikeinterface. The ratio is NaN for an empty cluster."""
    samples = drop_duplicated_spikes(spike_samples, sample_rate)
    n_spikes = len(samples)
    if n_spikes == 0 or duration <= 0:
        return np.nan, 0, n_spikes

    # Same float maths as spikeinterface (spike times in seconds), so a spike
    # pair right at the threshold is counted the same way in both.
    times = samples / sample_rate
    isi_threshold_s = isi_threshold_ms / 1000
    count = int(np.sum(np.diff(times) < isi_threshold_s))

    violation_rate = count / (2 * n_spikes * isi_threshold_s)
    total_rate = n_spikes / duration
    return violation_rate / total_rate, count, n_spikes
