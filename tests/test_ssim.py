import unittest
import numpy as np
from modules.pairwise_feature_lab import SSIMExtractor, PairwiseFeatureConfig

class TestSSIMExtractor(unittest.TestCase):
    def test_ssim_identical(self):
        img_a = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img_b = np.ones((100, 100, 3), dtype=np.uint8) * 128
        config = PairwiseFeatureConfig(
            hist_bins=64,
            hist_method="Correlation",
            color_mode="RGB",
            hist_grid_size=2,
            edge_blur="None",
            canny_low=50,
            canny_high=150,
            edge_grid_size=2,
            ssim_win_size=7,
            ssim_gaussian=False,
            text_thresh=127,
            text_kernel=5,
            text_iterations=1,
            text_min_area=10
        )
        extractor = SSIMExtractor()
        cache = {}
        logs = []
        result = extractor.extract(img_a, img_b, config, cache, logs)
        self.assertIn("ssim_mean", result.pairwise_metrics)
        self.assertAlmostEqual(result.pairwise_metrics["ssim_mean"], 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
