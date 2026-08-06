import unittest
import numpy as np
from modules.pairwise_feature_lab import HistogramExtractor, PairwiseFeatureConfig

class TestHistogramExtractor(unittest.TestCase):
    def test_basic_extraction(self):
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((100, 100, 3), dtype=np.uint8)
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
            text_min_area=10,
            hist_epsilon=1e-10
        )
        extractor = HistogramExtractor()
        cache = {}
        logs = []
        result = extractor.extract(img_a, img_b, config, cache, logs)
        self.assertIn("rgb_hist_dist_global_correlation", result.pairwise_metrics)
        self.assertAlmostEqual(result.pairwise_metrics["rgb_hist_dist_global_correlation"], 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
