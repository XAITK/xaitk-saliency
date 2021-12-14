import numpy as np
from typing import Iterable, Tuple, Dict, Hashable

from xaitk_saliency.interfaces.perturb_image import PerturbImage
from xaitk_saliency.interfaces.gen_detector_prop_sal import GenerateDetectorProposalSaliency
from xaitk_saliency.utils.tools.gen_detector_sal import gen_detector_sal

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox


class TestGenDetectorSal:

    class TestDetector(DetectImageObjects):
        """
        Dummy detector that returns constant detections
        """

        def detect_objects(
            self,
            img_iter: Iterable[np.ndarray]
        ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
            for i, img in enumerate(img_iter):
                yield [(AxisAlignedBoundingBox((0, 0), (100, 100)), {0: 1.0})] * i

        def get_config(self) -> dict: ...

    class TestPerturber(PerturbImage):
        """
        Dummy perturber that returns four masks, each with different
        corners of the image occluded.
        """

        def perturb(
            self,
            ref_image: np.ndarray
        ) -> np.ndarray:
            img_h = ref_image.shape[0]
            img_w = ref_image.shape[1]

            masks = np.ones((4, img_h, img_w))
            masks[0, 0:img_h//2, 0:img_w//2] = 0
            masks[1, 0:img_h//2, img_w//2:img_w] = 0
            masks[2, img_h//2:img_h, 0:img_w//2] = 0
            masks[3, img_h//2:img_h, img_w//2:img_w] = 0

            return masks

        def get_config(self) -> dict: ...

    class TestSalGenerator(GenerateDetectorProposalSaliency):
        """
        Dummy detection saliency generator. Combines perturbed detections
        and reference detection in an arbitrary way to produce maps.
        """

        def generate(
            self,
            ref_dets: np.ndarray,
            perturb_dets: np.ndarray,
            perturb_masks: np.ndarray,
        ) -> np.ndarray:

            m = [
                pert_det.sum() * mask
                for pert_det, mask in zip(perturb_dets, perturb_masks)
            ]

            sal_maps = np.asarray([
                ref_det.sum() * m[i]
                for i, ref_det in enumerate(ref_dets)
            ])

            sal_maps = sal_maps / sal_maps.max()

            return sal_maps

        def get_config(self) -> dict: ...

    def test_sal_generation(self) -> None:
        """
        Test saliency generation functionality with dummy images, detector,
        perturber, and saliency generator.
        """

        test_imgs = [np.full([10, 10, 3], fill_value=123, dtype=np.uint8)] * 3
        test_detector = self.TestDetector()
        test_perturber = self.TestPerturber()
        test_sal_gen = self.TestSalGenerator()

        sal_maps, dets = gen_detector_sal(test_imgs, test_detector, test_perturber, test_sal_gen, verbose=True)

        assert all([np.allclose(det, np.asarray([0, 0, 100, 100])) for det in dets])
        assert all([np.allclose(exp_map, sal_map) for exp_map, sal_map in zip(EXP_MAPS, sal_maps)])

    def test_sal_generation_one_dim(self) -> None:
        """
        Test saliency generation functionality with dummy images, detector,
        perturber, and saliency generator with grayscale images.
        """

        test_imgs = [np.full([10, 10], fill_value=123, dtype=np.uint8)] * 3
        test_detector = self.TestDetector()
        test_perturber = self.TestPerturber()
        test_sal_gen = self.TestSalGenerator()

        sal_maps, dets = gen_detector_sal(test_imgs, test_detector, test_perturber, test_sal_gen, verbose=True)

        assert all([np.allclose(det, np.asarray([0, 0, 100, 100])) for det in dets])
        assert all([np.allclose(exp_map, sal_map) for exp_map, sal_map in zip(EXP_MAPS, sal_maps)])


EXP_MAPS = [
    np.array([], dtype=np.float64).reshape(0, 10, 10),
    np.array([
        [
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ]
    ]),
    np.array([
        [
            [0., 0., 0., 0., 0., 0.05714286, 0.05714286, 0.05714286, 0.05714286, 0.05714286],
            [0., 0., 0., 0., 0., 0.05714286, 0.05714286, 0.05714286, 0.05714286, 0.05714286],
            [0., 0., 0., 0., 0., 0.05714286, 0.05714286, 0.05714286, 0.05714286, 0.05714286],
            [0., 0., 0., 0., 0., 0.05714286, 0.05714286, 0.05714286, 0.05714286, 0.05714286],
            [0., 0., 0., 0., 0., 0.05714286, 0.05714286, 0.05714286, 0.05714286, 0.05714286],
            [
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286
            ],
            [
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286
            ],
            [
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286
            ],
            [
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286
            ],
            [
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286,
                0.05714286
            ],
        ],
        [
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        ]
    ])
]
