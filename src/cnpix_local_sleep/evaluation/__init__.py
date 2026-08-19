"""Method-agnostic and cross-method evaluation of OFF-period detections.

Manual ("ground truth") OFF labels are shared infrastructure for *every*
detection method (morphological, sam3, harding, unit_based). This package holds
the label loaders, instance-label QC, image-stack grid geometry and pixel/event
metric kernels, plus the two drivers that score one detector against *another
detector's* output rather than against manual labels: ``head_to_head`` and
``banded_vs_morphological``.

Drivers that score a single method against the manual labels live with that
method: ``cnpix_local_sleep.morphological.manual_validation`` and
``.full48h_eval``, ``cnpix_local_sleep.unit_based.banded_eval``.

Light to import: ``metrics``, ``labels``, ``config``. The drivers that read
traces/parquets (``grid``, ``rasterize``, ``head_to_head``,
``banded_vs_morphological``) are imported as submodules on demand, e.g.::

    from cnpix_local_sleep.evaluation import head_to_head

SAM3 model labels are scored by ``samoffs``, which builds on the same kernels
via ``cnpix.evaluation``. It does not import ``cnpix_local_sleep``; the dependency runs
the other way.
"""

from cnpix_local_sleep.evaluation import config, labels, metrics

__all__ = ["config", "labels", "metrics"]
