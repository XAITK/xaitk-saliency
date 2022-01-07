from click.testing import CliRunner
import os
import py
import pytest
from unittest.mock import patch

from tests import DATA_DIR

from xaitk_saliency.utils.bin.sal_on_coco_dets import sal_on_coco_dets
from xaitk_saliency.exceptions import MismatchedLabelsError

from importlib.util import find_spec

deps = ['kwcoco']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dets_file = os.path.join(DATA_DIR, 'test_dets.json')
config_file = os.path.join(DATA_DIR, 'config.json')


class TestSalOnCocoDetsNotUsable:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_warning(self, tmpdir: py.path.local) -> None:
        """
        Test that proper warning is displayed when required dependencies are
        not installed.
        """

        output_dir = tmpdir.join('out')

        runner = CliRunner()

        if is_usable:
            with patch('xaitk_saliency.utils.bin.sal_on_coco_dets.is_usable', False):
                result = runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file)])
        else:
            result = runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file)])

        assert result.output == "This tool requires additional dependencies, please install 'xaitk-saliency[tools]'\n"
        assert not output_dir.check(dir=1)


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
class TestSalOnCocoDets:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_coco_sal_gen(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring.
        """

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file), "-v"])

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [4, 5]]

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_coco_sal_gen_img_overlay(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring with the overlay image option.
        """

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file), "--overlay-image"])

        # expected created directories for image saliency maps
        img_dirs = [output_dir.join(d) for d in ["test_image1", "test_image2"]]
        # detection ids that belong to each image
        img_dets = [[1, 2, 3], [4, 5]]

        assert sorted(output_dir.listdir()) == sorted(img_dirs)
        for img_dir, det_ids in zip(img_dirs, img_dets):
            map_files = [img_dir.join(f"det_{det_id}.jpeg") for det_id in det_ids]
            assert sorted(img_dir.listdir()) == sorted(map_files)

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """
        Test the generate configuration file option.
        """

        output_dir = tmpdir.join('out')

        output_config = tmpdir.join('gen_conf.json')

        runner = CliRunner()
        runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file), "-g", str(output_config)])

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir.check(dir=1)

    def test_mismatched_labels(self, tmpdir: py.path.local) -> None:
        """
        Test that exception is thrown when provided detections and detector
        have mismatched class labels.
        """

        dets_file = os.path.join(DATA_DIR, 'mismatched_dets.json')

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(sal_on_coco_dets, [str(dets_file), str(output_dir), str(config_file), "-v"])

        assert isinstance(result.exception, MismatchedLabelsError)
        assert not output_dir.check(dir=1)
