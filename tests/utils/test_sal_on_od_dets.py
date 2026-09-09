import json
import os
from pathlib import Path

import py
import pytest
from click.testing import CliRunner

from tests import DATA_DIR
from xaitk_saliency.utils.bin import sal_on_od_dets as sal_on_od_dets_cmd

config_file = os.path.join(DATA_DIR, "config.json")


@pytest.mark.tools
class TestSalOnOdDets:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_coco_sal_gen(self, tmpdir: py.path.local) -> None:
        """Test saliency map generation with RandomDetector, RISEGrid, and DRISEScoring."""
        output_dir = tmpdir.join(Path("out"))

        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets_cmd,
            [str(DATA_DIR), str(output_dir), str(config_file), "-o", "annotation_file=test_dets.json", "-v"],
        )

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(Path(d)) for d in ["test_image1", "test_image2"]]
        # index-based filenames are per-image positional, not original annotation ids
        img_dets = [[0, 1, 2], [0, 1]]

        assert result.exit_code == 0
        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets, strict=False):
            map_files = [img_dir.join(Path(f"det_{det_id}.jpeg")) for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_coco_sal_gen_img_overlay(self, tmpdir: py.path.local) -> None:
        """Test saliency map generation with RandomDetector, RISEGrid, and DRISEScoring with overlay image option."""
        output_dir = tmpdir.join(Path("out"))

        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets_cmd,
            [
                str(DATA_DIR),
                str(output_dir),
                str(config_file),
                "-o",
                "annotation_file=test_dets.json",
                "--overlay-image",
            ],
        )

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(Path(d)) for d in ["test_image1", "test_image2"]]
        # saliency-index-based filenames are per-image positional, not original annotation ids
        img_dets = [[0, 1, 2], [0, 1]]

        assert result.exit_code == 0
        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets, strict=False):
            map_files = [img_dir.join(Path(f"det_{det_id}.jpeg")) for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_invalid_dataset_format(self, tmpdir: py.path.local) -> None:
        """Check that click rejects an unrecognized --dataset-format value."""
        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets_cmd,
            [str(DATA_DIR), str(tmpdir), str(config_file), "-i", "none"],
        )

        assert result.exit_code == 2
        assert "'none' is not one of" in result.output

    def test_non_od_dataset_format(self, tmpdir: py.path.local) -> None:
        """Check that click rejects a dataset format that is not an object detection format."""
        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets_cmd,
            [str(DATA_DIR), str(tmpdir), str(config_file), "-i", "tao"],
        )

        assert result.exit_code == 2
        assert "'tao' is not one of" in result.output

    def test_invalid_loader_option(self, tmpdir: py.path.local) -> None:
        """Check that a malformed -o/--loader-option value is rejected."""
        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets_cmd,
            [str(DATA_DIR), str(tmpdir), str(config_file), "-o", "none"],
        )

        assert result.exit_code == 2
        assert "expected KEY=VALUE" in result.output

    def test_missing_annotations(self, tmpdir: py.path.local) -> None:
        """Check that an exception is raised when no annotation file is found in the dataset directory."""
        empty_dir = tmpdir.mkdir("empty")

        runner = CliRunner()
        with pytest.raises(ValueError, match=r"No categories declared"):
            runner.invoke(
                sal_on_od_dets_cmd,
                [str(empty_dir), str(tmpdir.join(Path("out"))), str(config_file)],
                catch_exceptions=False,
            )

    def test_no_categories(self, tmpdir: py.path.local) -> None:
        """Check that an exception is raised when the dataset has no declared category taxonomy."""
        empty_dir = tmpdir.mkdir("empty")
        (empty_dir / "annotations.json").write(
            json.dumps({"categories": [], "images": [], "annotations": []}),
        )

        runner = CliRunner()
        with pytest.raises(ValueError, match=r"No categories declared"):
            runner.invoke(
                sal_on_od_dets_cmd,
                [str(empty_dir), str(tmpdir.join(Path("out"))), str(config_file)],
                catch_exceptions=False,
            )
