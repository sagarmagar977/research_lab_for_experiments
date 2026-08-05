import unittest
import json
from modules.pairwise_feature_lab import FrameFeatures, PairwiseFeatures, PairwiseFeatureConfig
from modules.pairwise_feature_lab import CSVExporter, MarkdownExporter

class TestExporters(unittest.TestCase):
    def test_csv_exporter(self):
        fa = FrameFeatures(brightness=10.0, contrast=5.0, entropy=3.0, edge_density=0.1, text_occupancy=0.2)
        fb = FrameFeatures(brightness=12.0, contrast=6.0, entropy=3.1, edge_density=0.11, text_occupancy=0.21)
        pf = PairwiseFeatures(
            rgb_hist_dist_global=0.9,
            gray_hist_dist_global=0.85,
            rgb_hist_grid_mean=0.8,
            rgb_hist_grid_max=0.9,
            rgb_hist_grid_min=0.7,
            rgb_hist_grid_var=0.01,
            rgb_hist_grid_std=0.1,
            gray_hist_grid_mean=0.75,
            gray_hist_grid_max=0.85,
            gray_hist_grid_min=0.65,
            gray_hist_grid_var=0.02,
            gray_hist_grid_std=0.14,
            whole_edge_density_diff=0.01,
            grid_edge_mean_diff=0.02,
            grid_edge_max_diff=0.05,
            grid_edge_min_diff=0.0,
            grid_edge_var_diff=0.001,
            grid_edge_std_diff=0.03,
            ssim_mean=0.95,
            ssim_min=0.8,
            ssim_variance=0.005,
            mean_absolute_difference=2.5,
            text_occupancy_diff=0.01
        )
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
        csv_row = CSVExporter.export("fa.png", "fb.png", fa, fb, pf, config)
        self.assertTrue(csv_row.startswith("fa.png,fb.png,2.0.0,2.0.0,1.0.0"))
        
        md_report = MarkdownExporter.export(fa, fb, pf, config)
        self.assertIn("Pairwise Feature Vector Experiment Report", md_report)
        self.assertIn("MAD", md_report)

if __name__ == "__main__":
    unittest.main()
