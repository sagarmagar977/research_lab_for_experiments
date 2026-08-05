import unittest
import numpy as np
from modules.pairwise_feature_lab import EdgeExtractor, PairwiseFeatureConfig

class TestEdgeExtractor(unittest.TestCase):
    def test_edges_blank(self):
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
            text_min_area=10
        )
        extractor = EdgeExtractor()
        cache = {}
        logs = []
        result = extractor.extract(img_a, img_b, config, cache, logs)
        self.assertIn("edge_density", result.frame_a_metrics)
        self.assertEqual(result.frame_a_metrics["edge_density"], 0.0)

if __name__ == "__main__":
    unittest.main()
