import unittest
import numpy as np
from modules.pairwise_feature_lab import MorphologyExtractor, PairwiseFeatureConfig

class TestMorphologyExtractor(unittest.TestCase):
    def test_text_occupancy_blank(self):
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
        extractor = MorphologyExtractor()
        cache = {}
        logs = []
        result = extractor.extract(img_a, img_b, config, cache, logs)
        self.assertIn("text_occupancy", result.frame_a_metrics)
        # Inverted thresholding on black (0) with thresh=127 results in white (255) pixels since 0 < 127
        # Wait, if img is all 0 (black), THRESH_BINARY_INV turns all pixels to 255.
        # Connected components filter then finds one large component of size 10000 >= 10.
        # So text occupancy ratio should be 1.0!
        self.assertEqual(result.frame_a_metrics["text_occupancy"], 1.0)

if __name__ == "__main__":
    unittest.main()
