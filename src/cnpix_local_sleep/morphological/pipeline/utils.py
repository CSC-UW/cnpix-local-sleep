def log_step(step: str, **kwargs) -> None:
    """Helper to standardize logging output for processing steps.

    Args:
        step: The step name (e.g., "RUNNING", "PASSING", "DONE")
        **kwargs: Key-value pairs to include in the log message
    """
    details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"{step}: {details}")
