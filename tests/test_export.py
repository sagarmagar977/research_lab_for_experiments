import unittest
import json
import numpy as np
import cv2
from modules.pairwise_feature_lab import FrameFeatures, PairwiseFeatures, PairwiseFeatureConfig
from modules.pairwise_feature_lab import CSVExporter, HistogramExtractor

class TestExportersAndHistograms(unittest.TestCase):
    def test_csv_exporter(self):
        fa = FrameFeatures(
            brightness=10.0, contrast=5.0, entropy=3.0, edge_density=0.1, text_occupancy=0.2,
            global_rgb_hist_mean=0.1, global_rgb_hist_max=0.5, global_rgb_hist_min=0.0, global_rgb_hist_var=0.02, global_rgb_hist_std=0.14,
            global_gray_hist_mean=0.1, global_gray_hist_max=0.5, global_gray_hist_min=0.0, global_gray_hist_var=0.02, global_gray_hist_std=0.14,
            grid_rgb_hist_mean=0.1, grid_rgb_hist_max=0.5, grid_rgb_hist_min=0.0, grid_rgb_hist_var=0.02, grid_rgb_hist_std=0.14,
            grid_gray_hist_mean=0.1, grid_gray_hist_max=0.5, grid_gray_hist_min=0.0, grid_gray_hist_var=0.02, grid_gray_hist_std=0.14,
            grid_edge_mean=0.1, grid_edge_max=0.3, grid_edge_min=0.0, grid_edge_var=0.01, grid_edge_std=0.1
        )
        fb = FrameFeatures(
            brightness=12.0, contrast=6.0, entropy=3.1, edge_density=0.11, text_occupancy=0.21,
            global_rgb_hist_mean=0.12, global_rgb_hist_max=0.52, global_rgb_hist_min=0.01, global_rgb_hist_var=0.022, global_rgb_hist_std=0.142,
            global_gray_hist_mean=0.12, global_gray_hist_max=0.52, global_gray_hist_min=0.01, global_gray_hist_var=0.022, global_gray_hist_std=0.142,
            grid_rgb_hist_mean=0.12, grid_rgb_hist_max=0.52, grid_rgb_hist_min=0.01, grid_rgb_hist_var=0.022, grid_rgb_hist_std=0.142,
            grid_gray_hist_mean=0.12, grid_gray_hist_max=0.52, grid_gray_hist_min=0.01, grid_gray_hist_var=0.022, grid_gray_hist_std=0.142,
            grid_edge_mean=0.11, grid_edge_max=0.31, grid_edge_min=0.01, grid_edge_var=0.011, grid_edge_std=0.102
        )
        
        # Instantiate a PairwiseFeatures mock with 48 histogram properties
        pf_kwargs = {
            "rgb_hist_dist_global_correlation": 0.9,
            "rgb_hist_dist_global_intersection": 0.8,
            "rgb_hist_dist_global_bhattacharyya": 0.1,
            "rgb_hist_dist_global_chisquare": 0.2,
            "gray_hist_dist_global_correlation": 0.85,
            "gray_hist_dist_global_intersection": 0.75,
            "gray_hist_dist_global_bhattacharyya": 0.15,
            "gray_hist_dist_global_chisquare": 0.25,
            "whole_edge_density_diff": 0.01,
            "grid_edge_mean_diff": 0.02,
            "grid_edge_max_diff": 0.05,
            "grid_edge_min_diff": 0.0,
            "grid_edge_var_diff": 0.001,
            "grid_edge_std_diff": 0.03,
            "ssim_mean": 0.95,
            "ssim_min": 0.8,
            "ssim_variance": 0.005,
            "mean_absolute_difference": 2.5,
            "text_occupancy_diff": 0.01
        }
        
        # Populate all 4 grid metrics (Correlation, Intersection, Bhattacharyya, ChiSquare) for RGB and Grayscale
        for metric in ["correlation", "intersection", "bhattacharyya", "chisquare"]:
            for scale in ["rgb_hist_grid", "gray_hist_grid"]:
                pf_kwargs[f"{scale}_mean_{metric}"] = 0.5
                pf_kwargs[f"{scale}_max_{metric}"] = 0.8
                pf_kwargs[f"{scale}_min_{metric}"] = 0.2
                pf_kwargs[f"{scale}_var_{metric}"] = 0.05
                pf_kwargs[f"{scale}_std_{metric}"] = 0.22
                
        pf = PairwiseFeatures(**pf_kwargs)
        
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
        csv_row_default = CSVExporter.export("fa.png", "fb.png", fa, fb, pf, config)
        parts_default = csv_row_default.split(",")
        
        self.assertTrue(csv_row_default.startswith("fa.png,fb.png,2.0.0,2.0.0,1.0.0"))
        self.assertEqual(parts_default[14], "0")
        self.assertTrue("hist_bins" in parts_default[13])
        self.assertTrue(";" in parts_default[13])  # Verify commas replaced with semicolons

        csv_row_target = CSVExporter.export("fa.png", "fb.png", fa, fb, pf, config, ground_truth=1)
        parts_target = csv_row_target.split(",")
        self.assertEqual(parts_target[14], "1")

    def test_histogram_mathematical_assertions(self):
        extractor = HistogramExtractor()
        
        # Create identical images (e.g. constant value 128)
        img_a = np.full((100, 100, 3), 128, dtype=np.uint8)
        img_b = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # 1. Test identical images comparison
        h_a = extractor.calc_norm_hist(img_a, bins=32, color_mode="RGB", epsilon=1e-10)
        h_b = extractor.calc_norm_hist(img_b, bins=32, color_mode="RGB", epsilon=1e-10)
        
        comps = extractor.compare_hist_all(h_a, h_b)
        self.assertAlmostEqual(comps["correlation"], 1.0, places=4)
        self.assertAlmostEqual(comps["intersection"], 1.0, places=4)
        self.assertAlmostEqual(comps["bhattacharyya"], 0.0, places=4)
        self.assertAlmostEqual(comps["chisquare"], 0.0, places=4)
        
        # 2. Test empty/blank image (zeros)
        img_blank = np.zeros((100, 100, 3), dtype=np.uint8)
        h_blank = extractor.calc_norm_hist(img_blank, bins=32, color_mode="RGB", epsilon=1e-10)
        self.assertTrue(np.isfinite(h_blank).all())
        self.assertAlmostEqual(np.sum(h_blank), 1.0, places=5)
        
        # 3. Test brightness-shifted image
        img_shifted = np.clip(img_a.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        h_shifted = extractor.calc_norm_hist(img_shifted, bins=32, color_mode="RGB", epsilon=1e-10)
        comps_shifted = extractor.compare_hist_all(h_a, h_shifted)
        # Shifted image shape should be similar but shifted: correlation should be high but less than 1.0
        self.assertLess(comps_shifted["correlation"], 1.0)
        self.assertGreater(comps_shifted["bhattacharyya"], 0.0)
        self.assertGreater(comps_shifted["chisquare"], 0.0)
        
        # 4. Test completely different (white vs black)
        img_white = np.full((100, 100, 3), 255, dtype=np.uint8)
        h_white = extractor.calc_norm_hist(img_white, bins=32, color_mode="RGB", epsilon=1e-10)
        comps_diff = extractor.compare_hist_all(h_blank, h_white)
        
        self.assertLess(comps_diff["correlation"], 0.5)
        # Maximum Bhattacharyya distance should be near 1.0 because probability mass is concentrated at opposite bins
        self.assertGreater(comps_diff["bhattacharyya"], 0.9)
        self.assertGreater(comps_diff["chisquare"], 10.0)
        
        # 5. Test Random Noise comparison
        np.random.seed(42)
        img_noise = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        h_noise = extractor.calc_norm_hist(img_noise, bins=32, color_mode="RGB", epsilon=1e-10)
        comps_noise = extractor.compare_hist_all(h_a, h_noise)
        
        # Noise vs flat image should exhibit low correlation, large bhattacharyya, and large chisquare
        self.assertLess(comps_noise["correlation"], 0.2)
        self.assertGreater(comps_noise["bhattacharyya"], 0.5)
        self.assertGreater(comps_noise["chisquare"], 1.0)

if __name__ == "__main__":
    unittest.main()
