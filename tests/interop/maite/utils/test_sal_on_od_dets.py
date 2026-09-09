import json
import os
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import py
import pytest
from click.testing import CliRunner
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

from tests import DATA_DIR
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack
from xaitk_saliency.interop.maite.utils.bin import sal_on_od_dets

rng = np.random.default_rng()

dataset_dir_path = DATA_DIR
config_file = os.path.join(DATA_DIR, "config.json")


@pytest.mark.maite
@pytest.mark.tools
class TestSalOnOdDets:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    mock_return_value = (
        [
            rng.integers(low=0, high=255, size=(3, 256, 256), dtype=np.uint8),
            rng.integers(low=0, high=255, size=(3, 256, 256), dtype=np.uint8),
        ],
        {
            "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
            "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
                "n": 10,
                "s": 8,
                "p1": 0.5,
                "seed": 0,
                "threads": 4,
            },
        },
    )

    @pytest.mark.parametrize("overlay_image", [False, True])
    @mock.patch(
        "xaitk_saliency.interop.maite.utils.bin._sal_on_od_dets.compute_sal_maps",
        return_value=mock_return_value,
    )
    def test_compute_sal_maps(
        self,
        compute_sal_maps_patch: MagicMock,
        overlay_image: bool,
        tmpdir: py.path.local,
    ) -> None:
        """Test that compute_sal_maps is called appropriately and the images are saved correctly."""
        output_dir_path = tmpdir.join(Path("out"))

        runner = CliRunner()
        runner_args = [str(dataset_dir_path), str(output_dir_path), str(config_file), "-v"]
        if overlay_image:
            runner_args.append("--overlay-image")
        result = runner.invoke(sal_on_od_dets, runner_args, catch_exceptions=False)

        # Confirm compute_sal_maps arguments are as expected
        kwargs = compute_sal_maps_patch.call_args.kwargs
        assert len(kwargs["dataset"]) == 2
        assert isinstance(kwargs["sal_generator"], DRISEStack)
        assert isinstance(kwargs["blackbox_detector"], RandomDetector)
        assert kwargs["num_classes"] == 3

        # expected created directories for image saliency maps
        img_dir_paths = [output_dir_path.join(Path(d)) for d in ["test_image1", "test_image2"]]
        # index-based filenames are per-image positional, not original annotation ids
        img_dets = [[0, 1, 2], [0, 1, 2]]

        assert result.exit_code == 0
        assert sorted(output_dir_path.listdir()) == sorted(img_dir_paths)
        for img_dir, det_ids in zip(img_dir_paths, img_dets, strict=False):
            map_files = [img_dir.join(Path(f"det_{det_id}.jpeg")) for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    @mock.patch(
        "xaitk_saliency.interop.maite.utils.bin._sal_on_od_dets.compute_sal_maps",
        return_value=mock_return_value,
    )
    def test_prob_detection_scored_by_max_prob(self, compute_sal_maps_patch: MagicMock, tmpdir: py.path.local) -> None:
        """Test that a prob-only detection is scored by its max probability."""
        runner = CliRunner()
        runner.invoke(
            sal_on_od_dets,
            [str(dataset_dir_path), str(tmpdir.join(Path("out"))), str(config_file)],
            catch_exceptions=False,
        )

        dataset = compute_sal_maps_patch.call_args.kwargs["dataset"]
        scores = {det.source_annotation_id: det.score for det in dataset.samples[1].detections}
        assert scores[6] == pytest.approx(0.5)

    @mock.patch(
        "xaitk_saliency.interop.maite.utils.bin._sal_on_od_dets.compute_sal_maps",
        return_value=mock_return_value,
    )
    def test_detection_without_score_defaults_to_one(
        self,
        compute_sal_maps_patch: MagicMock,
        tmpdir: py.path.local,
    ) -> None:
        """Test that a detection with no score or prob is kept with a score of 1.0."""
        runner = CliRunner()
        runner.invoke(
            sal_on_od_dets,
            [str(dataset_dir_path), str(tmpdir.join(Path("out"))), str(config_file)],
            catch_exceptions=False,
        )

        dataset = compute_sal_maps_patch.call_args.kwargs["dataset"]
        scores = {det.source_annotation_id: det.score for det in dataset.samples[1].detections}
        assert scores[4] == pytest.approx(1.0)

    def test_invalid_dataset_format(self, tmpdir: py.path.local) -> None:
        """Check that click rejects an unrecognized --dataset-format value."""
        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets,
            [str(dataset_dir_path), str(tmpdir), str(config_file), "-i", "none"],
        )

        assert result.exit_code == 2
        assert "'none' is not one of" in result.output

    def test_non_od_dataset_format(self, tmpdir: py.path.local) -> None:
        """Check that click rejects a dataset format that is not an object detection format."""
        runner = CliRunner()
        result = runner.invoke(
            sal_on_od_dets,
            [str(dataset_dir_path), str(tmpdir), str(config_file), "-i", "tao"],
        )

        assert result.exit_code == 2
        assert "'tao' is not one of" in result.output

    def test_missing_annotations(self, tmpdir: py.path.local) -> None:
        """Check that an exception is raised when no annotation file is found in the dataset directory."""
        empty_dir = tmpdir.mkdir("empty")

        runner = CliRunner()
        with pytest.raises(ValueError, match=r"No categories declared"):
            runner.invoke(
                sal_on_od_dets,
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
                sal_on_od_dets,
                [str(empty_dir), str(tmpdir.join(Path("out"))), str(config_file)],
                catch_exceptions=False,
            )
