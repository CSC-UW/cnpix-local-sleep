---
title: cnpix-local-sleep TODO
updated: 2026-10-12
---

# cnpix-local-sleep TODO

- ~~Test the stack writer on zarr 3 and ome-zarr 0.12, then lift the pin.~~
  Done 2026-10-10: the workspace moved to zarr 3 and `ome-zarr>=0.12`;
  `cnpix_local_sleep.stacks.write` now writes the single-scale store itself
  and keeps the dataset-`0` / zarr v2 layout the readers expect. Record:
  `gfys_workspace/docs/developer_notes/zarr3_brainglobe3_migration.md`.
