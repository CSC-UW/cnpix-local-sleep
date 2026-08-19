"""Type definitions for the morphological detection pipeline."""

from typing import Literal, TypedDict


class NdImageFilterKwargs(TypedDict, total=False):
    """Arguments for ndimage filter operations.

    Attributes:
        size: Tuple of (n_samples, n_channels) for filter size
        sigma: Tuple of (n_samples, n_channels) for Gaussian sigma
    """

    size: tuple[int, int]
    sigma: tuple[float, float]


class DetectionOpts(TypedDict, total=False):
    """Options for OFF period detection.

    Bundles all algorithm configuration for OFF detection into a single dict,
    replacing the previous pattern of passing 8+ individual parameters. Loaded
    from the YAML template under ``cnpix_local_sleep.morphological.mua.data/`` (see helpers
    in ``cnpix_local_sleep.morphological.detection_opts``), or constructed programmatically.

    Attributes:
        threshold_method: Method for computing thresholds. Only
            ``"from_value"`` remains: read the quantile from the method's
            thresholds CSV.
        ndimage_filter_type: Type of ndimage filter to apply (e.g.,
            "median", None)
        ndimage_filter_kwargs: Keyword arguments for the ndimage filter
        clean_binary_mask: Whether to apply morphological cleaning to
            binary mask
        n_channels_clean: Number of channels for morphological cleaning
            operations
        n_channels_connect: Number of channels for connecting across bad
            channels
        n_samples_clean: Number of samples for removing shorter blobs
        n_samples_connect: Number of samples for connecting across time
            (None to skip)
    """

    threshold_method: Literal["from_value"]
    ndimage_filter_type: str | None
    ndimage_filter_kwargs: NdImageFilterKwargs | None
    clean_binary_mask: bool
    n_channels_clean: int | None
    n_channels_connect: int | None
    n_samples_clean: int | None
    n_samples_connect: int | None


_YAML_TEMPLATE_KEYS = list(DetectionOpts.__annotations__)
"""Keys expected in YAML detection opts templates."""


def validate_detection_opts(opts: DetectionOpts | None) -> None:
    """Validate that DetectionOpts contains required keys and valid values.

    Args:
        opts: Detection options dictionary to validate

    Raises:
        ValueError: If opts is None, missing required keys, or contains
            invalid values
    """
    if opts is None:
        raise ValueError("opts parameter is required")

    missing_keys = [k for k in _YAML_TEMPLATE_KEYS if k not in opts]
    if missing_keys:
        raise ValueError(f"opts missing required keys: {missing_keys}")

    valid_methods = ["from_value"]
    if opts.get("threshold_method") not in valid_methods:
        raise ValueError(
            f"Invalid threshold_method '{opts.get('threshold_method')}'. "
            f"Must be one of {valid_methods}"
        )
