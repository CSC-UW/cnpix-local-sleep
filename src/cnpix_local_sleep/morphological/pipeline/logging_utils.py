"""Logging utilities for the unit-free OFF detection pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from typing import TextIO


class PipelineLogger:
    """Logger that writes timestamped start/end entries for pipeline steps.

    Parameters
    ----------
    log_dir
        Directory where log file will be created. If None, logging is disabled.
    enabled
        Whether logging is enabled. If False, all operations are no-ops.

    Examples
    --------
    >>> logger = PipelineLogger(Path("/tmp/logs"))
    >>> with logger.step("Processing data"):
    ...     # do work
    ...     pass
    >>> logger.close()
    """

    log_path: Path | None
    enabled: bool
    _file: TextIO | None

    def __init__(self, log_dir: Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled and log_dir is not None
        if self.enabled and log_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = log_dir / f"pipeline_run_{timestamp}.log"
        else:
            self.log_path = None
        self._file = None

    def log(self, message: str) -> None:
        """Write a timestamped message to the log file."""
        if not self.enabled or self.log_path is None:
            return
        if self._file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_path, "w")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"[{timestamp}] {message}\n")
        self._file.flush()

    @contextmanager
    def step(self, name: str) -> Generator[None, None, None]:
        """Context manager that logs the start and end of a named step.

        Parameters
        ----------
        name
            Name of the step to log.

        Yields
        ------
        None
        """
        self.log(f"START: {name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(f"END: {name} (elapsed: {elapsed:.2f}s)")

    def close(self) -> None:
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None