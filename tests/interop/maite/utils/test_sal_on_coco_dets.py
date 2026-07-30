import logging
import os
import unittest.mock as mock
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest
from click.testing import CliRunner
from smqtk_detection.impls.detect_image_objects.random_detector import RandomDetector

from tests import DATA_DIR
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack
from xaitk_saliency.interop.maite.utils.bin.sal_on_coco_dets import sal_on_coco_dets

rng = np.random.default_rng()

deps = ["kwcoco", "matplotlib"]
specs = [find_spec(dep) for dep in deps]
is_usable = all(spec is not None for spec in specs)

dataset_dir_path = DATA_DIR
config_file = os.path.join(DATA_DIR, "config.json")


class TestSalOnCocoDetsNotUsable:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    @mock.patch("xaitk_saliency.interop.maite.utils.bin.sal_on_coco_dets.is_usable", False)
    def test_warning(self, tmpdir: py.path.local) -> None:
        """Test that proper warning is displayed when required dependencies are not installed."""
        output_dir_path = tmpdir.join(Path("out"))

        runner = CliRunner()

        result = runner.invoke(sal_on_coco_dets, [str(dataset_dir_path), str(output_dir_path), str(config_file)])

        assert result.output.startswith(
            "This tool requires additional dependencies, please install 'xaitk-saliency[tools]'.",
        )
        assert not output_dir_path.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
class TestSalOnCocoDets:
    """These tests make use of the `tmpdir` fixture from `pytest`.

    Find more information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    mock_return_value = (
        [
            rng.integers(0, 255, (3, 256, 256), dtype=np.uint8),
            rng.integers(0, 255, (3, 256, 256), dtype=np.uint8),
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

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """Test the generate configuration file option."""
        output_dir_path = tmpdir.join(Path("out"))

        output_config = tmpdir.join(Path("gen_conf.json"))

        runner = CliRunner()
        runner.invoke(
            sal_on_coco_dets,
            [
                str(dataset_dir_path),
                str(output_dir_path),
                str(config_file),
                "-g",
                str(output_config),
            ],
        )

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir_path.check(dir=1)

    @pytest.mark.parametrize("overlay_image", [False, True])
    @mock.patch(
        "xaitk_saliency.interop.maite.utils.bin.sal_on_coco_dets.compute_sal_maps",
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
        result = runner.invoke(sal_on_coco_dets, runner_args, catch_exceptions=False)

        # Confirm compute_sal_maps arguments are as expected
        kwargs = compute_sal_maps_patch.call_args.kwargs
        assert len(kwargs["dataset"]) == 2
        assert isinstance(kwargs["sal_generator"], DRISEStack)
        assert isinstance(kwargs["blackbox_detector"], RandomDetector)
        assert kwargs["num_classes"] == 3

        # expected created directories for image saliency maps
        img_dir_paths = [output_dir_path.join(Path(d)) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [4, 5, 6]]

        assert result.exit_code == 0
        assert sorted(output_dir_path.listdir()) == sorted(img_dir_paths)
        for img_dir, det_ids in zip(img_dir_paths, img_dets, strict=False):
            map_files = [img_dir.join(Path(f"det_{det_id}.jpeg")) for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    @mock.patch("pathlib.Path.is_file", return_value=False)
    def test_missing_annotations(self, tmpdir: py.path.local) -> None:
        """Check that an exception is appropriately raised if the annotations file is missing."""
        with pytest.raises(ValueError, match=r"Could not identify annotations file."):
            CliRunner().invoke(
                sal_on_coco_dets,
                [str(dataset_dir_path), str(tmpdir), str(config_file), "-v"],
                catch_exceptions=False,
            )

    @mock.patch("pathlib.Path.is_file", side_effect=[True, False, False])
    @mock.patch(
        "xaitk_saliency.interop.maite.utils.bin.sal_on_coco_dets.compute_sal_maps",
        return_value=mock_return_value,
    )
    def test_missing_metadata(
        self,
        _: MagicMock,  # noqa: PT019
        is_file_patch: MagicMock,  # noqa:ARG002
        caplog: pytest.LogCaptureFixture,
        tmpdir: py.path.local,
    ) -> None:
        """Check that the entrypoint is able to continue when a metadata file is not present.

        This will only run as as long as it's not required by the perturber.
        """
        with caplog.at_level(logging.INFO):
            runner = CliRunner()
            result = runner.invoke(
                sal_on_coco_dets,
                [str(dataset_dir_path), str(tmpdir), str(config_file), "-v"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0

        assert "Could not identify metadata file, assuming no metadata." in caplog.text
