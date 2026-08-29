import unittest
import numpy as np
from modules.pairwise_feature_lab import MorphologyExtractor, PairwiseFeatureConfig

class TestMorphologyExtractor(unittest.TestCase):
    def test_text_occupancy_blank_dark(self):
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((100, 100, 3), dtype=np.uint8)
        config = PairwiseFeatureConfig(
            hist_bins=64, hist_method="Correlation", color_mode="RGB", hist_grid_size=2,
            edge_blur="None", canny_low=50, canny_high=150, edge_grid_size=2,
            ssim_win_size=7, ssim_gaussian=False, text_thresh=127, text_kernel=5,
            text_iterations=1, text_min_area=10
        )
        extractor = MorphologyExtractor()
        result = extractor.extract(img_a, img_b, config, {}, [])
        self.assertIn("text_occupancy", result.frame_a_metrics)
        # Blank dark image has no text -> 0.0
        self.assertEqual(result.frame_a_metrics["text_occupancy"], 0.0)

    def test_text_occupancy_dark_with_text(self):
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_a[40:60, 40:60] = 255 # White text on black background
        config = PairwiseFeatureConfig(
            hist_bins=64, hist_method="Correlation", color_mode="RGB", hist_grid_size=2,
            edge_blur="None", canny_low=50, canny_high=150, edge_grid_size=2,
            ssim_win_size=7, ssim_gaussian=False, text_thresh=127, text_kernel=5,
            text_iterations=1, text_min_area=10
        )
        extractor = MorphologyExtractor()
        result = extractor.extract(img_a, img_a, config, {}, [])
        # Image with text -> text_occupancy > 0.0
        self.assertGreater(result.frame_a_metrics["text_occupancy"], 0.0)

if __name__ == "__main__":
    unittest.main()
