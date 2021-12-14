import numpy as np
from PIL import Image  # type: ignore
import json
from click.testing import CliRunner
import py

from xaitk_saliency.utils.bin.det_sal_on_img_dir import det_sal_on_img_dir


class TestDetSalOnImgDir:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_sal_gen(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring.
        """

        imgs = [Image.fromarray(np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)) for _ in range(3)]

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        for i, img in enumerate(imgs):
            img.save(str(img_dir.join(f"test_{i}.png")))

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file), "-v"])

        # expected output directories
        expected_outputs = [output_dir.join(f"test_{i}") for i in range(3)]
        # sort to normalize order
        assert sorted(output_dir.listdir()) == sorted(expected_outputs)

    def test_sal_gen_one_dimension(self, tmpdir: py.path.local) -> None:
        """
        Test saliency map generation with RandomDetector, RISEGrid, and
        DRISEScoring on grayscale image.
        """

        imgs = [Image.fromarray(np.random.randint(255, size=(100, 100), dtype=np.uint8)) for _ in range(3)]

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        for i, img in enumerate(imgs):
            img.save(str(img_dir.join(f"test_{i}.png")))

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file)])

        # expected output directories
        expected_outputs = [output_dir.join(f"test_{i}") for i in range(3)]
        # sort to normalize order
        assert sorted(output_dir.listdir()) == sorted(expected_outputs)

    def test_sal_gen_with_overlay(self, tmpdir: py.path.local) -> None:
        """
        Test normal functionality with overlaid image option.
        """

        imgs = [Image.fromarray(np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)) for _ in range(4)]

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        for i, img in enumerate(imgs):
            img.save(str(img_dir.join(f"test_{i}.png")))

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file), "--overlay-image"])

        # expected output directories
        expected_outputs = [output_dir.join(f"test_{i}") for i in range(4)]
        # sort to normalize order
        assert sorted(output_dir.listdir()) == sorted(expected_outputs)

    def test_config_gen(self, tmpdir: py.path.local) -> None:
        """
        Test the generate configuration file option.
        """

        imgs = [Image.fromarray(np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)) for _ in range(2)]

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        for i, img in enumerate(imgs):
            img.save(str(img_dir.join(f"test_{i}.png")))

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        output_config = tmpdir.join('config.txt')

        runner = CliRunner()
        runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file), "-g", str(output_config)])

        # check that config file was created
        assert output_config.check(file=1)
        # check that no output was generated
        assert not output_dir.check(dir=1)

    def test_missing_img_dir(self, tmpdir: py.path.local) -> None:
        """
        Test that correct error message is output when image directory is
        missing.
        """

        img_dir = tmpdir.join('imgs')

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file)])

        error_msg = ''.join((
            "Usage: det-sal-on-img-dir [OPTIONS] IMG_DIR OUTPUT_DIR CONFIG_FILE\n",
            "Try 'det-sal-on-img-dir -h' for help.\n",
            "\n"
            f"Error: Invalid value for 'IMG_DIR': Directory '{str(img_dir)}' does not exist.\n"
        ))

        assert result.output == error_msg
        # check that no output was generated
        assert not output_dir.check(dir=1)

    def test_missing_conf_file(self, tmpdir: py.path.local) -> None:
        """
        Test that correct error message is output when config file is missing.
        """

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        config_file = tmpdir.join('conf.txt')

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file)])

        error_msg = ''.join((
            "Usage: det-sal-on-img-dir [OPTIONS] IMG_DIR OUTPUT_DIR CONFIG_FILE\n",
            "Try 'det-sal-on-img-dir -h' for help.\n",
            "\n"
            f"Error: Invalid value for 'CONFIG_FILE': '{str(config_file)}': No such file or directory\n"
        ))

        assert result.output == error_msg
        # check that no output was generated
        assert not output_dir.check(dir=1)

    def test_invalid_img(self, tmpdir: py.path.local) -> None:
        """
        Test that output is not generated for non-image files.
        """

        imgs = [Image.fromarray(np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)) for _ in range(2)]

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        for i, img in enumerate(imgs):
            img.save(str(img_dir.join(f"test_{i}.png")))

        invalid_img = img_dir.join('test.png')
        invalid_img.write_text("This is not an image file.", 'utf-8')

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        result = runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file)])

        # expected output directories, which does not include the invalid file
        expected_outputs = [output_dir.join(f"test_{i}") for i in range(2)]

        assert result.output == "File test.png is not a valid image file.\n"
        # sort to normalize order
        assert sorted(output_dir.listdir()) == sorted(expected_outputs)

    def test_image_without_dot(self, tmpdir: py.path.local) -> None:
        """
        Test that output directory names are generated correctly when image
        file has no file extension.
        """

        img = Image.fromarray(np.random.randint(255, size=(100, 100, 3), dtype=np.uint8))

        img_dir = tmpdir.join('imgs')
        img_dir.mkdir()

        img.save(str(img_dir.join("test_0")), format='png')

        config_file = tmpdir.join('conf.txt')

        with config_file.open('w') as outfile:
            json.dump(TEST_CONFIG, outfile)

        output_dir = tmpdir.join('out')

        runner = CliRunner()
        runner.invoke(det_sal_on_img_dir, [str(img_dir), str(output_dir), str(config_file)])

        # expected output directory
        expected_output = [output_dir.join("test_0")]
        assert output_dir.listdir() == expected_output


TEST_CONFIG = {
    "DetectImageObjects": {
        "type": "smqtk_detection.impls.detect_image_objects.random_detector.RandomDetector",
                "smqtk_detection.impls.detect_image_objects.random_detector.RandomDetector": {},
    },
    "PerturbImage": {
        "type": "xaitk_saliency.impls.perturb_image.rise.RISEGrid",
                "xaitk_saliency.impls.perturb_image.rise.RISEGrid": {
                    "n": 10,
                    "s": 8,
                    "p1": 0.5,
                    "seed": 0,
                    "threads": 4
                },
    },
    "GenerateDetectorProposalSaliency": {
        "type": "xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring.DRISEScoring",
                "xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring.DRISEScoring": {
                    "proximity_metric": "cosine"
                }
    }
}
