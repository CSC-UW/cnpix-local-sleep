from cnpix_local_sleep.stacks import files as stk


def test_stack_paths_live_in_samoffs_without_method_level():
    p = stk.get_sam3_off_stacks_ome_zarr_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    assert p.name == "off_stacks.ome.zarr"
    assert p.parts[-3:-1] == ("probe=imec0", "condition=Early.REC.NREM")
    assert "method=sam3" not in p.parts
    assert "samoffs" in p.parts


def test_timestamps_beside_stack():
    z = stk.get_sam3_off_stacks_ome_zarr_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    t = stk.get_sam3_off_stacks_timestamps_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    assert t.parent == z.parent and t.name == "timestamps.zarr"


def test_savedir_matches_reader_dir():
    z = stk.get_sam3_off_stacks_ome_zarr_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    s = stk.get_sam3_savedir_path("CNPIX2-Segundo", "imec0", "Early.REC.NREM")
    assert s == z.parent
