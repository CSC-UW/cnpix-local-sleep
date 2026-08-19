"""The OFF tables published as GitHub Release assets, and how to fetch them.

``r-offp/inst/extdata`` is split in two, and the line is *whether the manuscript's
tables need the file offline* rather than file size (the two happen to coincide):

* The 24 summarized tables are committed (1.31 MB). Every reported model reads
  only those, so ``renv::install('.')`` plus ``make_manuscript_tables.sh`` rebuilds
  the supplement from a bare clone with no network and no credentials.
* The three event-level tables (``full48h_{llas,clas,blas}_offs.parquet``, 347 MB
  together) are Release assets. One of them exceeds GitHub's 100 MB per-file
  limit, and only the Supplementary Figure S2 / edge-adjusted path reads them.

The Release carries *both* halves, so one asset set is a complete, self-contained
copy of the data for archiving or a later move to Zenodo. Only the three are ever
fetched, because the other 24 are already in the checkout.

Nothing here runs on a machine that has run the exporters:
:func:`get_event_table_path` prefers a file already sitting in ``inst/extdata`` and
falls back to the cache only when there is none.
"""

import datetime
import json
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

REPO = "CSC-UW/cnpix-local-sleep"
"""Repository whose Releases host the data."""

TAG = "latest"
"""Release tag. Rolling: a re-export replaces the assets in place and the tag moves to
the current commit, which is what should happen while the numbers are still moving. The
cost is that a clone does not pin its own data; pin it at acceptance by cutting a fixed
tag and repointing this constant. See ``docs/DATA.md``.
"""

EVENT_LEVEL_ASSETS = (
    "full48h_llas_offs.parquet",
    "full48h_clas_offs.parquet",
    "full48h_blas_offs.parquet",
)
"""The tables that are not committed, and so may have to be fetched."""

CACHE_ENV_VAR = "CNPIX_LOCAL_SLEEP_CACHE"


def asset_url(name: str, repo: str = REPO, tag: str = TAG) -> str:
    """The public download URL for one Release asset. Needs no authentication."""
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def cache_dir(tag: str = TAG) -> pathlib.Path:
    """Where fetched assets are cached, keyed by release tag.

    Keyed by tag so that repointing at another tag re-fetches rather than silently
    reusing whatever ``latest`` happened to hold. Honours
    ``$CNPIX_LOCAL_SLEEP_CACHE``; otherwise the platform cache directory. Never
    ``inst/extdata`` -- an installed package directory is often read-only, is shared,
    and is wiped by the next ``renv::install('.')``.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        root = pathlib.Path(override).expanduser()
    elif platform.system() == "Darwin":
        root = pathlib.Path.home() / "Library" / "Caches" / "cnpix_local_sleep"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = pathlib.Path(xdg).expanduser() if xdg else pathlib.Path.home() / ".cache"
        root = base / "cnpix_local_sleep"
    return root / "release_data" / tag


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _remote_mtime(name: str, repo: str, tag: str) -> float | None:
    """Asset ``updatedAt`` as a POSIX timestamp, or ``None`` if it cannot be read.

    Best-effort: this is what makes a re-uploaded (``--clobber``) asset invalidate an
    existing cache entry. Any failure -- no ``gh``, no network, no such release --
    means "cannot tell", and the cached copy is then used as-is rather than the call
    failing.
    """
    if not _gh_available():
        return None
    try:
        out = subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        for asset in json.loads(out).get("assets", []):
            if asset.get("name") == name:
                stamp = asset.get("updatedAt")
                if not stamp:
                    return None
                parsed = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                return parsed.timestamp()
    except (subprocess.SubprocessError, OSError, ValueError, KeyError):
        return None
    return None


def _download(name: str, dest: pathlib.Path, repo: str, tag: str) -> None:
    """Fetch one asset to ``dest``, atomically.

    ``gh`` when it is on PATH, plain HTTPS otherwise, so no ``gh`` install is needed.
    Downloads land on a temporary sibling and are renamed into place, so an interrupted
    212 MB transfer cannot leave a truncated file that looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    if _gh_available():
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            try:
                subprocess.run(
                    ["gh", "release", "download", tag, "--repo", repo,
                     "--pattern", name, "--dir", tmp],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                pathlib.Path(tmp, name).replace(dest)
                return
            except subprocess.CalledProcessError as exc:
                errors.append(f"gh: {(exc.stderr or '').strip() or exc}")
            except OSError as exc:
                errors.append(f"gh: {exc}")

    url = asset_url(name, repo=repo, tag=tag)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp_path.replace(dest)
        return
    except (urllib.error.URLError, OSError) as exc:
        errors.append(f"https: {exc}")
        tmp_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not fetch {name!r} from the {tag!r} release of {repo}.\n"
        + "\n".join(f"  {e}" for e in errors)
        + f"\n\nDownload it by hand and put it in {dest.parent} (or in "
        f"r-offp/inst/extdata):\n  {url}"
    )


def get_event_table_path(
    dataset: str,
    extdata_dir: pathlib.Path | None = None,
    repo: str = REPO,
    tag: str = TAG,
) -> pathlib.Path:
    """Resolve one event-level OFF table, fetching it only if it is not local.

    Parameters
    ----------
    dataset
        One of ``"llas"``, ``"clas"``, ``"blas"``.
    extdata_dir
        Checkout location to prefer. Defaults to ``r-offp/inst/extdata``. A file
        there always wins, so a machine that has run ``off-analysis
        export-full48h-offs`` never touches the network.

    Returns
    -------
    Path to a readable parquet, either in the checkout or in the cache.
    """
    name = f"full48h_{dataset}_offs.parquet"
    if name not in EVENT_LEVEL_ASSETS:
        raise ValueError(
            f"{dataset!r} is not an event-level dataset; expected one of "
            f"{[a.split('_')[1] for a in EVENT_LEVEL_ASSETS]}"
        )

    if extdata_dir is None:
        from cnpix_local_sleep import files

        extdata_dir = files.get_r_offp_extdata_dir()
    local = pathlib.Path(extdata_dir) / name
    if local.exists():
        return local

    cached = cache_dir(tag) / name
    if cached.exists():
        remote = _remote_mtime(name, repo, tag)
        if remote is None or remote <= cached.stat().st_mtime:
            return cached

    _download(name, cached, repo, tag)
    return cached


def is_available(
    dataset: str, extdata_dir: pathlib.Path | None = None, tag: str = TAG
) -> bool:
    """Whether an event-level table can be read without downloading it.

    Lets a caller (a test, a notebook) decide to skip rather than silently pull
    347 MB. Does not touch the network.
    """
    name = f"full48h_{dataset}_offs.parquet"
    if extdata_dir is None:
        from cnpix_local_sleep import files

        extdata_dir = files.get_r_offp_extdata_dir()
    return (pathlib.Path(extdata_dir) / name).exists() or (
        cache_dir(tag) / name
    ).exists()


def publish(
    extdata_dir: pathlib.Path,
    repo: str = REPO,
    tag: str = TAG,
    event_level_only: bool = False,
) -> list[str]:
    """Upload ``inst/extdata`` to the Release, creating it if it does not exist.

    Uploads every file in ``extdata_dir`` by default, not just the three that have to
    be hosted: the committed 24 are cheap and their presence makes one asset set a
    complete copy of the data. ``--clobber`` replaces assets in place, which is what
    invalidates :func:`get_event_table_path`'s cache.

    Returns the asset names uploaded.
    """
    extdata_dir = pathlib.Path(extdata_dir)
    if event_level_only:
        paths = [extdata_dir / name for name in EVENT_LEVEL_ASSETS]
    else:
        # Dotfiles are not data. A stray .DS_Store would otherwise become a
        # permanent asset on what becomes a public release.
        paths = sorted(
            p
            for p in extdata_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Nothing to publish; run the exporters first. Missing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    if not _gh_available():
        raise RuntimeError("`gh` is required to publish; see https://cli.github.com")

    exists = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        subprocess.run(
            ["gh", "release", "create", tag, "--repo", repo,
             "--title", f"Data ({tag})",
             "--notes", (
                 "OFF tables for *Local sleep during wake: A Neuropixels view*.\n\n"
                 "The three `full48h_*_offs.parquet` event-level tables are hosted "
                 "here because one exceeds GitHub's 100 MB per-file limit. The "
                 "summarized tables are also attached for completeness; they are "
                 "committed in `r-offp/inst/extdata/` and need no download."
             )],
            check=True,
        )

    subprocess.run(
        ["gh", "release", "upload", tag, *[str(p) for p in paths],
         "--repo", repo, "--clobber"],
        check=True,
    )
    return [p.name for p in paths]
