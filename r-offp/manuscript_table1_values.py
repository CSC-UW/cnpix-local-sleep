"""Manuscript Table 1: OFF counts per subject and structure across the six conditions.

29 subject-structure pairs x 6 conditions, blocked by the three OFF partitions
(All OFFs / Medium + Large / Small OFFs; 522 values), plus late SD as a percentage of
early SD (blank where early SD has no OFF periods). The counts are the ``count``
column of ``inst/extdata/summarized_full48h_*_offs.parquet``, the files the
Supplementary Table S1 models are fit on.

A (subject, structure, condition) cell absent from an OFF parquet is a Wake condition
in which no OFF period was detected under that criterion, so its count is 0.

Subjects are shown with the anonymized IDs used across the cnpix_local_sleep publication
figures (Subject1..Subject15; see ``subject_aliases``); conditions use the external
names (Early/Late baseline, Early/Late SD, Early/Late recovery).

Usage:
    python manuscript_table1_values.py             # -> _output_manuscript/manuscript_table1_values.{xlsx,csv}
    python manuscript_table1_values.py --xlsx out.xlsx --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXTDATA = HERE / "inst" / "extdata"

# The three OFF partitions in the manuscript, in S1 column order (no BLAS).
SETS = OrderedDict([
    ("llas", "All OFFs"),
    ("clas", "Medium + Large"),
    ("llas_exclusive", "Small OFFs"),
])

# Six conditions in experimental-time order. Internal tokens are used only for
# lookups against the parquets; the sheet and CSV show the external names.
CONDITIONS = [
    ("Early.BSL.NREM", "Early baseline"),
    ("Early.REC.NREM.Match", "Late baseline"),
    ("Early.NOD.Wake", "Early SD"),
    ("Late.NOD.Wake", "Late SD"),
    ("Early.REC.NREM", "Early recovery"),
    ("Late.REC.NREM", "Late recovery"),
]
CONDITION_ORDER = [c for c, _ in CONDITIONS]
CONDITION_LABEL = dict(CONDITIONS)  # internal token -> external name

TITLE = "OFF count"
SUBTITLE = "Counts (number of OFF periods), per subject and structure across the six conditions."
SD_RATIO_HEADER = "Late SD as % of early SD"
SD_RATIO_AFTER = CONDITION_ORDER.index("Late.NOD.Wake") + 1


def subject_aliases(subjects) -> dict:
    """Canonical anonymized IDs used across the cnpix_local_sleep publication figures:
    Subject1..SubjectN by lexicographic sort of the internal CNPIX* IDs (so
    'CNPIX10-Charles' -> 'Subject1', 'CNPIX2-Segundo' -> 'Subject10'). Mirrors
    ``{s: f"Subject{i}" for i, s in enumerate(sorted(offs.subject.unique()), 1)}``
    in notebooks/figures/*."""
    return {s: f"Subject{i}" for i, s in enumerate(sorted(set(subjects)), start=1)}


def _round_sig(x, sig: int = 4):
    """Round to ``sig`` significant figures, returning a float (so Excel stores a
    real number)."""
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def load_counts():
    """Return {dataset: [(subject, structure, [count per condition])]} for the three
    OFF partitions, over the sorted (subject, structure) pairs present in any of them."""
    counts = {}
    for ds in SETS:
        df = pd.read_parquet(EXTDATA / f"summarized_full48h_{ds}_offs.parquet")
        counts[ds] = {
            (r.subject, r.structure, r.condition): int(r.count)
            for r in df[["subject", "structure", "condition", "count"]].itertuples(index=False)
        }
    combos = sorted({(s, st) for c in counts.values() for s, st, _ in c})
    return {
        ds: [
            (subject, structure, [c.get((subject, structure, cond), 0) for cond in CONDITION_ORDER])
            for subject, structure in combos
        ]
        for ds, c in counts.items()
    }


def with_sd_ratio(vals):
    """Insert late SD as a percentage of early SD after the Late SD column."""
    early = vals[CONDITION_ORDER.index("Early.NOD.Wake")]
    late = vals[CONDITION_ORDER.index("Late.NOD.Wake")]
    ratio = _round_sig(100 * late / early) if early else None
    return vals[:SD_RATIO_AFTER] + [ratio] + vals[SD_RATIO_AFTER:]


def write_xlsx(counts, anon, path: Path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    thin = Side(style="thin", color="999999")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    fill = PatternFill("solid", fgColor="E8E8E8")

    headers = ["Subject", "Structure"] + [plain for _tok, plain in CONDITIONS]
    headers.insert(2 + SD_RATIO_AFTER, SD_RATIO_HEADER)
    ncol = len(headers)

    wb = Workbook()
    ws = wb.active
    ws.title = TITLE
    ws.cell(1, 1, TITLE).font = Font(bold=True, size=12)
    sc = ws.cell(2, 1, SUBTITLE)
    sc.font = Font(italic=True, size=9)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
    sc.alignment = left
    hdr_row = 3
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(hdr_row, c, h)
        cell.font = Font(bold=True)
        cell.alignment = center
        cell.border = Border(bottom=thin, top=thin)
    r = hdr_row + 1

    for ds, rows in counts.items():
        bc = ws.cell(r, 1, SETS[ds])
        bc.font = Font(bold=True, italic=True)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=ncol)
        for c in range(1, ncol + 1):
            ws.cell(r, c).fill = fill
        r += 1
        for subject, structure, vals in rows:
            ws.cell(r, 1, anon[subject]).alignment = left
            ws.cell(r, 2, structure).alignment = left
            for j, v in enumerate(with_sd_ratio(vals)):
                ws.cell(r, 3 + j, v).alignment = center
            r += 1

    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 12
    for i in range(ncol - 2):
        ws.column_dimensions[chr(ord("C") + i)].width = 15
    ws.freeze_panes = f"C{hdr_row + 1}"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_csv(counts, anon, path: Path):
    """Tidy long form: one row per (partition, subject, structure, condition), with
    anonymized subject IDs and external condition names (no internal identifiers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["off_partition", "subject", "structure", "condition", "off_count"])
        for ds, rows in counts.items():
            for subject, structure, vals in rows:
                for cond, v in zip(CONDITION_ORDER, vals, strict=True):
                    w.writerow([SETS[ds], anon[subject], structure, CONDITION_LABEL[cond], v])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "_output_manuscript")
    args = parser.parse_args(argv)

    counts = load_counts()
    anon = subject_aliases(s for rows in counts.values() for s, _, _ in rows)

    xlsx = args.xlsx or (args.out_dir / "manuscript_table1_values.xlsx")
    csv_path = args.csv or (args.out_dir / "manuscript_table1_values.csv")
    write_xlsx(counts, anon, xlsx)
    write_csv(counts, anon, csv_path)
    print(f"n combos: {len(next(iter(counts.values())))}")
    print(f"XLSX: {xlsx}\nCSV : {csv_path}")


if __name__ == "__main__":
    main()
