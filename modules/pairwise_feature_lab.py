import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from dataclasses import dataclass, asdict
import time
from abc import ABC, abstractmethod
import json
import datetime

# ==========================================
# 1. Configuration & Data Models
# ==========================================

@dataclass
class PairwiseFeatureConfig:
    hist_bins: int
    hist_method: str
    color_mode: str
    hist_grid_size: int
    edge_blur: str
    canny_low: int
    canny_high: int
    edge_grid_size: int
    ssim_win_size: int
    ssim_gaussian: bool
    text_thresh: int
    text_kernel: int
    text_iterations: int
    text_min_area: int
    hist_epsilon: float = 1e-10

@dataclass
class FrameFeatures:
    brightness: float
    contrast: float
    entropy: float
    edge_density: float
    text_occupancy: float
    
    # Global Histogram Stats
    global_rgb_hist_mean: float
    global_rgb_hist_max: float
    global_rgb_hist_min: float
    global_rgb_hist_var: float
    global_rgb_hist_std: float
    global_gray_hist_mean: float
    global_gray_hist_max: float
    global_gray_hist_min: float
    global_gray_hist_var: float
    global_gray_hist_std: float
    
    # Grid Histogram Stats
    grid_rgb_hist_mean: float
    grid_rgb_hist_max: float
    grid_rgb_hist_min: float
    grid_rgb_hist_var: float
    grid_rgb_hist_std: float
    grid_gray_hist_mean: float
    grid_gray_hist_max: float
    grid_gray_hist_min: float
    grid_gray_hist_var: float
    grid_gray_hist_std: float
    
    # Grid Edge Stats
    grid_edge_mean: float
    grid_edge_max: float
    grid_edge_min: float
    grid_edge_var: float
    grid_edge_std: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_list(self) -> list:
        return [
            self.brightness, self.contrast, self.entropy, self.edge_density, self.text_occupancy,
            self.global_rgb_hist_mean, self.global_rgb_hist_max, self.global_rgb_hist_min, self.global_rgb_hist_var, self.global_rgb_hist_std,
            self.global_gray_hist_mean, self.global_gray_hist_max, self.global_gray_hist_min, self.global_gray_hist_var, self.global_gray_hist_std,
            self.grid_rgb_hist_mean, self.grid_rgb_hist_max, self.grid_rgb_hist_min, self.grid_rgb_hist_var, self.grid_rgb_hist_std,
            self.grid_gray_hist_mean, self.grid_gray_hist_max, self.grid_gray_hist_min, self.grid_gray_hist_var, self.grid_gray_hist_std,
            self.grid_edge_mean, self.grid_edge_max, self.grid_edge_min, self.grid_edge_var, self.grid_edge_std
        ]

@dataclass
class PairwiseFeatures:
    # Global RGB
    rgb_hist_dist_global_correlation: float
    rgb_hist_dist_global_intersection: float
    rgb_hist_dist_global_bhattacharyya: float
    rgb_hist_dist_global_chisquare: float

    # Global Grayscale
    gray_hist_dist_global_correlation: float
    gray_hist_dist_global_intersection: float
    gray_hist_dist_global_bhattacharyya: float
    gray_hist_dist_global_chisquare: float

    # Grid RGB - Correlation
    rgb_hist_grid_mean_correlation: float
    rgb_hist_grid_max_correlation: float
    rgb_hist_grid_min_correlation: float
    rgb_hist_grid_var_correlation: float
    rgb_hist_grid_std_correlation: float

    # Grid RGB - Intersection
    rgb_hist_grid_mean_intersection: float
    rgb_hist_grid_max_intersection: float
    rgb_hist_grid_min_intersection: float
    rgb_hist_grid_var_intersection: float
    rgb_hist_grid_std_intersection: float

    # Grid RGB - Bhattacharyya
    rgb_hist_grid_mean_bhattacharyya: float
    rgb_hist_grid_max_bhattacharyya: float
    rgb_hist_grid_min_bhattacharyya: float
    rgb_hist_grid_var_bhattacharyya: float
    rgb_hist_grid_std_bhattacharyya: float

    # Grid RGB - ChiSquare
    rgb_hist_grid_mean_chisquare: float
    rgb_hist_grid_max_chisquare: float
    rgb_hist_grid_min_chisquare: float
    rgb_hist_grid_var_chisquare: float
    rgb_hist_grid_std_chisquare: float

    # Grid Grayscale - Correlation
    gray_hist_grid_mean_correlation: float
    gray_hist_grid_max_correlation: float
    gray_hist_grid_min_correlation: float
    gray_hist_grid_var_correlation: float
    gray_hist_grid_std_correlation: float

    # Grid Grayscale - Intersection
    gray_hist_grid_mean_intersection: float
    gray_hist_grid_max_intersection: float
    gray_hist_grid_min_intersection: float
    gray_hist_grid_var_intersection: float
    gray_hist_grid_std_intersection: float

    # Grid Grayscale - Bhattacharyya
    gray_hist_grid_mean_bhattacharyya: float
    gray_hist_grid_max_bhattacharyya: float
    gray_hist_grid_min_bhattacharyya: float
    gray_hist_grid_var_bhattacharyya: float
    gray_hist_grid_std_bhattacharyya: float

    # Grid Grayscale - ChiSquare
    gray_hist_grid_mean_chisquare: float
    gray_hist_grid_max_chisquare: float
    gray_hist_grid_min_chisquare: float
    gray_hist_grid_var_chisquare: float
    gray_hist_grid_std_chisquare: float

    # Non-histogram features
    whole_edge_density_diff: float
    grid_edge_mean_diff: float
    grid_edge_max_diff: float
    grid_edge_min_diff: float
    grid_edge_var_diff: float
    grid_edge_std_diff: float
    ssim_mean: float
    ssim_min: float
    ssim_variance: float
    mean_absolute_difference: float
    text_occupancy_diff: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_list(self) -> list:
        return [
            # Global RGB
            self.rgb_hist_dist_global_correlation,
            self.rgb_hist_dist_global_intersection,
            self.rgb_hist_dist_global_bhattacharyya,
            self.rgb_hist_dist_global_chisquare,

            # Global Grayscale
            self.gray_hist_dist_global_correlation,
            self.gray_hist_dist_global_intersection,
            self.gray_hist_dist_global_bhattacharyya,
            self.gray_hist_dist_global_chisquare,

            # Grid RGB - Correlation
            self.rgb_hist_grid_mean_correlation,
            self.rgb_hist_grid_max_correlation,
            self.rgb_hist_grid_min_correlation,
            self.rgb_hist_grid_var_correlation,
            self.rgb_hist_grid_std_correlation,

            # Grid RGB - Intersection
            self.rgb_hist_grid_mean_intersection,
            self.rgb_hist_grid_max_intersection,
            self.rgb_hist_grid_min_intersection,
            self.rgb_hist_grid_var_intersection,
            self.rgb_hist_grid_std_intersection,

            # Grid RGB - Bhattacharyya
            self.rgb_hist_grid_mean_bhattacharyya,
            self.rgb_hist_grid_max_bhattacharyya,
            self.rgb_hist_grid_min_bhattacharyya,
            self.rgb_hist_grid_var_bhattacharyya,
            self.rgb_hist_grid_std_bhattacharyya,

            # Grid RGB - ChiSquare
            self.rgb_hist_grid_mean_chisquare,
            self.rgb_hist_grid_max_chisquare,
            self.rgb_hist_grid_min_chisquare,
            self.rgb_hist_grid_var_chisquare,
            self.rgb_hist_grid_std_chisquare,

            # Grid Grayscale - Correlation
            self.gray_hist_grid_mean_correlation,
            self.gray_hist_grid_max_correlation,
            self.gray_hist_grid_min_correlation,
            self.gray_hist_grid_var_correlation,
            self.gray_hist_grid_std_correlation,

            # Grid Grayscale - Intersection
            self.gray_hist_grid_mean_intersection,
            self.gray_hist_grid_max_intersection,
            self.gray_hist_grid_min_intersection,
            self.gray_hist_grid_var_intersection,
            self.gray_hist_grid_std_intersection,

            # Grid Grayscale - Bhattacharyya
            self.gray_hist_grid_mean_bhattacharyya,
            self.gray_hist_grid_max_bhattacharyya,
            self.gray_hist_grid_min_bhattacharyya,
            self.gray_hist_grid_var_bhattacharyya,
            self.gray_hist_grid_std_bhattacharyya,

            # Grid Grayscale - ChiSquare
            self.gray_hist_grid_mean_chisquare,
            self.gray_hist_grid_max_chisquare,
            self.gray_hist_grid_min_chisquare,
            self.gray_hist_grid_var_chisquare,
            self.gray_hist_grid_std_chisquare,

            # Non-histogram
            self.whole_edge_density_diff,
            self.grid_edge_mean_diff,
            self.grid_edge_max_diff,
            self.grid_edge_min_diff,
            self.grid_edge_var_diff,
            self.grid_edge_std_diff,
            self.ssim_mean,
            self.ssim_min,
            self.ssim_variance,
            self.mean_absolute_difference,
            self.text_occupancy_diff
        ]

@dataclass
class VisualArtifacts:
    edges_a: np.ndarray
    edges_b: np.ndarray
    text_mask_a: np.ndarray
    text_mask_b: np.ndarray
    ssim_map: np.ndarray
    difference_map: np.ndarray
    changed_pixel_count: int
    changed_pixel_pct: float
    hist_grid_scores: list[float]
    edge_grid_scores: list[float]

@dataclass
class ExtractorResult:
    frame_a_metrics: dict
    frame_b_metrics: dict
    pairwise_metrics: dict
    visuals: dict

# ==========================================
# 2. Extractor Plugin Base Interface
# ==========================================

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, img_a: np.ndarray, img_b: np.ndarray, config: PairwiseFeatureConfig, cache: dict, logs: list) -> ExtractorResult:
        pass

# ==========================================
# 3. Concrete Extractor Plugins
# ==========================================

class HistogramExtractor(BaseExtractor):
    def get_grid_cells(self, shape, grid_size):
        h, w = shape[:2]
        cell_h = h / grid_size
        cell_w = w / grid_size
        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                ymin, ymax = int(i * cell_h), int(min((i + 1) * cell_h, h))
                xmin, xmax = int(j * cell_w), int(min((j + 1) * cell_w, w))
                cells.append((ymin, ymax, xmin, xmax))
        return cells

    def calc_norm_hist(self, img, bins, color_mode, epsilon):
        if color_mode == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        else:
            hists = []
            for ch in range(3):
                hists.append(cv2.calcHist([img], [ch], None, [bins], [0, 256]))
            hist = np.concatenate(hists, axis=0)
            
        hist = hist.astype(np.float32).flatten()
        hist += epsilon
        hist /= np.sum(hist)
        return hist

    def verify_histogram_validity(self, h1: np.ndarray, h2: np.ndarray):
        assert h1.dtype == np.float32 and h2.dtype == np.float32, "Histograms must be float32 for comparison"
        assert h1.ndim == 1 and h2.ndim == 1, "Histograms must be 1D"
        assert np.isfinite(h1).all() and np.isfinite(h2).all(), "Histograms must contain only finite values"
        assert (h1 >= 0.0).all() and (h2 >= 0.0).all(), "Histogram bins cannot be negative"
        assert abs(np.sum(h1) - 1.0) < 1e-5 and abs(np.sum(h2) - 1.0) < 1e-5, "Histograms must sum to 1.0 (L1 normalized)"

    def compare_hist_all(self, h1, h2):
        self.verify_histogram_validity(h1, h2)
        return {
            "correlation": float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)),
            "intersection": float(cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)),
            "bhattacharyya": float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)),
            "chisquare": float(cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR))
        }

    def extract(self, img_a: np.ndarray, img_b: np.ndarray, config: PairwiseFeatureConfig, cache: dict, logs: list) -> ExtractorResult:
        t0 = time.perf_counter()
        
        # Unique hashes for images to check cache
        hash_a = hash(img_a.tobytes())
        hash_b = hash(img_b.tobytes())
        
        # Grayscale conversions for entropy
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        # Calculate single frame metrics (entropy, brightness, contrast)
        def get_frame_metrics(img, gray, img_hash):
            ent_key = ("entropy", img_hash)
            if ent_key in cache:
                logs.append(f"[Histogram] Single frame metrics cache HIT")
                return cache[ent_key].copy()
                
            # Brightness (mean) and Contrast (std)
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Shannon Entropy
            hist_ent = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_ent = hist_ent / (hist_ent.sum() + 1e-7)
            entropy = float(-np.sum(hist_ent * np.log2(hist_ent + 1e-7)))
            
            res = {"brightness": brightness, "contrast": contrast, "entropy": entropy}
            cache[ent_key] = res
            return res.copy()

        fm_a = get_frame_metrics(img_a, gray_a, hash_a)
        fm_b = get_frame_metrics(img_b, gray_b, hash_b)
        
        # Global Histograms Cache logic
        hist_rgb_key_a = ("hist_rgb", hash_a, config.hist_bins, config.hist_epsilon)
        hist_rgb_key_b = ("hist_rgb", hash_b, config.hist_bins, config.hist_epsilon)
        hist_gray_key_a = ("hist_gray", hash_a, config.hist_bins, config.hist_epsilon)
        hist_gray_key_b = ("hist_gray", hash_b, config.hist_bins, config.hist_epsilon)
        
        if hist_rgb_key_a not in cache:
            cache[hist_rgb_key_a] = self.calc_norm_hist(img_a, config.hist_bins, "RGB", config.hist_epsilon)
        if hist_rgb_key_b not in cache:
            cache[hist_rgb_key_b] = self.calc_norm_hist(img_b, config.hist_bins, "RGB", config.hist_epsilon)
        if hist_gray_key_a not in cache:
            cache[hist_gray_key_a] = self.calc_norm_hist(img_a, config.hist_bins, "Grayscale", config.hist_epsilon)
        if hist_gray_key_b not in cache:
            cache[hist_gray_key_b] = self.calc_norm_hist(img_b, config.hist_bins, "Grayscale", config.hist_epsilon)
            
        rgb_hist_a = cache[hist_rgb_key_a]
        rgb_hist_b = cache[hist_rgb_key_b]
        gray_hist_a = cache[hist_gray_key_a]
        gray_hist_b = cache[hist_gray_key_b]
        
        # Compute global histogram stats for A
        fm_a["global_rgb_hist_mean"] = float(np.mean(rgb_hist_a))
        fm_a["global_rgb_hist_max"] = float(np.max(rgb_hist_a))
        fm_a["global_rgb_hist_min"] = float(np.min(rgb_hist_a))
        fm_a["global_rgb_hist_var"] = float(np.var(rgb_hist_a))
        fm_a["global_rgb_hist_std"] = float(np.std(rgb_hist_a))
        
        fm_a["global_gray_hist_mean"] = float(np.mean(gray_hist_a))
        fm_a["global_gray_hist_max"] = float(np.max(gray_hist_a))
        fm_a["global_gray_hist_min"] = float(np.min(gray_hist_a))
        fm_a["global_gray_hist_var"] = float(np.var(gray_hist_a))
        fm_a["global_gray_hist_std"] = float(np.std(gray_hist_a))
        
        # Compute global histogram stats for B
        fm_b["global_rgb_hist_mean"] = float(np.mean(rgb_hist_b))
        fm_b["global_rgb_hist_max"] = float(np.max(rgb_hist_b))
        fm_b["global_rgb_hist_min"] = float(np.min(rgb_hist_b))
        fm_b["global_rgb_hist_var"] = float(np.var(rgb_hist_b))
        fm_b["global_rgb_hist_std"] = float(np.std(rgb_hist_b))
        
        fm_b["global_gray_hist_mean"] = float(np.mean(gray_hist_b))
        fm_b["global_gray_hist_max"] = float(np.max(gray_hist_b))
        fm_b["global_gray_hist_min"] = float(np.min(gray_hist_b))
        fm_b["global_gray_hist_var"] = float(np.var(gray_hist_b))
        fm_b["global_gray_hist_std"] = float(np.std(gray_hist_b))
        
        rgb_global_comps = self.compare_hist_all(rgb_hist_a, rgb_hist_b)
        gray_global_comps = self.compare_hist_all(gray_hist_a, gray_hist_b)
        
        # Grid-based histograms
        cells_a = self.get_grid_cells(img_a.shape, config.hist_grid_size)
        cells_b = self.get_grid_cells(img_b.shape, config.hist_grid_size)
        
        grid_rgb_comps = {"correlation": [], "intersection": [], "bhattacharyya": [], "chisquare": []}
        grid_gray_comps = {"correlation": [], "intersection": [], "bhattacharyya": [], "chisquare": []}
        
        grid_rgb_means_a, grid_rgb_maxes_a, grid_rgb_mins_a, grid_rgb_vars_a, grid_rgb_stds_a = [], [], [], [], []
        grid_gray_means_a, grid_gray_maxes_a, grid_gray_mins_a, grid_gray_vars_a, grid_gray_stds_a = [], [], [], [], []
        
        grid_rgb_means_b, grid_rgb_maxes_b, grid_rgb_mins_b, grid_rgb_vars_b, grid_rgb_stds_b = [], [], [], [], []
        grid_gray_means_b, grid_gray_maxes_b, grid_gray_mins_b, grid_gray_vars_b, grid_gray_stds_b = [], [], [], [], []
        
        for idx, ((ya1, ya2, xa1, xa2), (yb1, yb2, xb1, xb2)) in enumerate(zip(cells_a, cells_b)):
            cell_a = img_a[ya1:ya2, xa1:xa2]
            cell_b = img_b[yb1:yb2, xb1:xb2]
            
            c_rgb_a = self.calc_norm_hist(cell_a, config.hist_bins, "RGB", config.hist_epsilon)
            c_rgb_b = self.calc_norm_hist(cell_b, config.hist_bins, "RGB", config.hist_epsilon)
            c_gray_a = self.calc_norm_hist(cell_a, config.hist_bins, "Grayscale", config.hist_epsilon)
            c_gray_b = self.calc_norm_hist(cell_b, config.hist_bins, "Grayscale", config.hist_epsilon)
            
            # Statistics of individual cells for A
            grid_rgb_means_a.append(np.mean(c_rgb_a))
            grid_rgb_maxes_a.append(np.max(c_rgb_a))
            grid_rgb_mins_a.append(np.min(c_rgb_a))
            grid_rgb_vars_a.append(np.var(c_rgb_a))
            grid_rgb_stds_a.append(np.std(c_rgb_a))
            
            grid_gray_means_a.append(np.mean(c_gray_a))
            grid_gray_maxes_a.append(np.max(c_gray_a))
            grid_gray_mins_a.append(np.min(c_gray_a))
            grid_gray_vars_a.append(np.var(c_gray_a))
            grid_gray_stds_a.append(np.std(c_gray_a))
            
            # Statistics of individual cells for B
            grid_rgb_means_b.append(np.mean(c_rgb_b))
            grid_rgb_maxes_b.append(np.max(c_rgb_b))
            grid_rgb_mins_b.append(np.min(c_rgb_b))
            grid_rgb_vars_b.append(np.var(c_rgb_b))
            grid_rgb_stds_b.append(np.std(c_rgb_b))
            
            grid_gray_means_b.append(np.mean(c_gray_b))
            grid_gray_maxes_b.append(np.max(c_gray_b))
            grid_gray_mins_b.append(np.min(c_gray_b))
            grid_gray_vars_b.append(np.var(c_gray_b))
            grid_gray_stds_b.append(np.std(c_gray_b))
            
            rgb_comps = self.compare_hist_all(c_rgb_a, c_rgb_b)
            gray_comps = self.compare_hist_all(c_gray_a, c_gray_b)
            
            for m in ["correlation", "intersection", "bhattacharyya", "chisquare"]:
                grid_rgb_comps[m].append(rgb_comps[m])
                grid_gray_comps[m].append(gray_comps[m])
            
        # Merge grid cell stats for A
        fm_a["grid_rgb_hist_mean"] = float(np.mean(grid_rgb_means_a))
        fm_a["grid_rgb_hist_max"] = float(np.mean(grid_rgb_maxes_a))
        fm_a["grid_rgb_hist_min"] = float(np.mean(grid_rgb_mins_a))
        fm_a["grid_rgb_hist_var"] = float(np.mean(grid_rgb_vars_a))
        fm_a["grid_rgb_hist_std"] = float(np.mean(grid_rgb_stds_a))
        
        fm_a["grid_gray_hist_mean"] = float(np.mean(grid_gray_means_a))
        fm_a["grid_gray_hist_max"] = float(np.mean(grid_gray_maxes_a))
        fm_a["grid_gray_hist_min"] = float(np.mean(grid_gray_mins_a))
        fm_a["grid_gray_hist_var"] = float(np.mean(grid_gray_vars_a))
        fm_a["grid_gray_hist_std"] = float(np.mean(grid_gray_stds_a))
        
        # Merge grid cell stats for B
        fm_b["grid_rgb_hist_mean"] = float(np.mean(grid_rgb_means_b))
        fm_b["grid_rgb_hist_max"] = float(np.mean(grid_rgb_maxes_b))
        fm_b["grid_rgb_hist_min"] = float(np.mean(grid_rgb_mins_b))
        fm_b["grid_rgb_hist_var"] = float(np.mean(grid_rgb_vars_b))
        fm_b["grid_rgb_hist_std"] = float(np.mean(grid_rgb_stds_b))
        
        fm_b["grid_gray_hist_mean"] = float(np.mean(grid_gray_means_b))
        fm_b["grid_gray_hist_max"] = float(np.mean(grid_gray_maxes_b))
        fm_b["grid_gray_hist_min"] = float(np.mean(grid_gray_mins_b))
        fm_b["grid_gray_hist_var"] = float(np.mean(grid_gray_vars_b))
        fm_b["grid_gray_hist_std"] = float(np.mean(grid_gray_stds_b))
        
        pairwise_res = {
            "rgb_hist_dist_global_correlation": rgb_global_comps["correlation"],
            "rgb_hist_dist_global_intersection": rgb_global_comps["intersection"],
            "rgb_hist_dist_global_bhattacharyya": rgb_global_comps["bhattacharyya"],
            "rgb_hist_dist_global_chisquare": rgb_global_comps["chisquare"],

            "gray_hist_dist_global_correlation": gray_global_comps["correlation"],
            "gray_hist_dist_global_intersection": gray_global_comps["intersection"],
            "gray_hist_dist_global_bhattacharyya": gray_global_comps["bhattacharyya"],
            "gray_hist_dist_global_chisquare": gray_global_comps["chisquare"]
        }
        
        # Build grid statistics for each metric
        for m in ["correlation", "intersection", "bhattacharyya", "chisquare"]:
            rgb_arr = np.array(grid_rgb_comps[m])
            gray_arr = np.array(grid_gray_comps[m])
            
            pairwise_res[f"rgb_hist_grid_mean_{m}"] = float(np.mean(rgb_arr))
            pairwise_res[f"rgb_hist_grid_max_{m}"] = float(np.max(rgb_arr))
            pairwise_res[f"rgb_hist_grid_min_{m}"] = float(np.min(rgb_arr))
            pairwise_res[f"rgb_hist_grid_var_{m}"] = float(np.var(rgb_arr))
            pairwise_res[f"rgb_hist_grid_std_{m}"] = float(np.std(rgb_arr))
            
            pairwise_res[f"gray_hist_grid_mean_{m}"] = float(np.mean(gray_arr))
            pairwise_res[f"gray_hist_grid_max_{m}"] = float(np.max(gray_arr))
            pairwise_res[f"gray_hist_grid_min_{m}"] = float(np.min(gray_arr))
            pairwise_res[f"gray_hist_grid_var_{m}"] = float(np.var(gray_arr))
            pairwise_res[f"gray_hist_grid_std_{m}"] = float(np.std(gray_arr))
        
        elapsed = (time.perf_counter() - t0) * 1000
        logs.append(f"[Histogram] Features computed in {elapsed:.1f} ms")
        
        return ExtractorResult(
            frame_a_metrics=fm_a,
            frame_b_metrics=fm_b,
            pairwise_metrics=pairwise_res,
            visuals={"hist_grid_scores": grid_rgb_comps["correlation"]} # fallback visual
        )

class EdgeExtractor(BaseExtractor):
    def get_canny_edges(self, img, blur_size_str, canny_low, canny_high):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if blur_size_str != "None":
            try:
                k = int(blur_size_str.split("x")[0])
                gray = cv2.GaussianBlur(gray, (k, k), 0)
            except Exception:
                pass
        edges = cv2.Canny(gray, canny_low, canny_high)
        return edges

    def get_grid_cells(self, shape, grid_size):
        h, w = shape[:2]
        cell_h = h / grid_size
        cell_w = w / grid_size
        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                ymin, ymax = int(i * cell_h), int(min((i + 1) * cell_h, h))
                xmin, xmax = int(j * cell_w), int(min((j + 1) * cell_w, w))
                cells.append((ymin, ymax, xmin, xmax))
        return cells

    def extract(self, img_a: np.ndarray, img_b: np.ndarray, config: PairwiseFeatureConfig, cache: dict, logs: list) -> ExtractorResult:
        t0 = time.perf_counter()
        
        hash_a = hash(img_a.tobytes())
        hash_b = hash(img_b.tobytes())
        
        edge_key_a = ("edges", hash_a, config.edge_blur, config.canny_low, config.canny_high)
        edge_key_b = ("edges", hash_b, config.edge_blur, config.canny_low, config.canny_high)
        
        # Canny edge maps caching
        if edge_key_a in cache:
            logs.append(f"[Edge] Frame A edges cache HIT")
            edges_a = cache[edge_key_a]
        else:
            edges_a = self.get_canny_edges(img_a, config.edge_blur, config.canny_low, config.canny_high)
            cache[edge_key_a] = edges_a
            
        if edge_key_b in cache:
            logs.append(f"[Edge] Frame B edges cache HIT")
            edges_b = cache[edge_key_b]
        else:
            edges_b = self.get_canny_edges(img_b, config.edge_blur, config.canny_low, config.canny_high)
            cache[edge_key_b] = edges_b
            
        density_a = float(np.sum(edges_a == 255) / edges_a.size)
        density_b = float(np.sum(edges_b == 255) / edges_b.size)
        
        whole_diff = abs(density_a - density_b)
        
        # Grid-wise Edge Density differences
        cells_a = self.get_grid_cells(edges_a.shape, config.edge_grid_size)
        cells_b = self.get_grid_cells(edges_b.shape, config.edge_grid_size)
        
        densities_a = []
        densities_b = []
        cell_diffs = []
        
        for (ya1, ya2, xa1, xa2), (yb1, yb2, xb1, xb2) in zip(cells_a, cells_b):
            cell_a = edges_a[ya1:ya2, xa1:xa2]
            cell_b = edges_b[yb1:yb2, xb1:xb2]
            
            c_dens_a = np.sum(cell_a == 255) / cell_a.size
            c_dens_b = np.sum(cell_b == 255) / cell_b.size
            
            densities_a.append(c_dens_a)
            densities_b.append(c_dens_b)
            cell_diffs.append(abs(c_dens_a - c_dens_b))
            
        cell_diffs = np.array(cell_diffs)
        densities_a = np.array(densities_a)
        densities_b = np.array(densities_b)
        
        fm_a = {
            "edge_density": density_a,
            "grid_edge_mean": float(np.mean(densities_a)),
            "grid_edge_max": float(np.max(densities_a)),
            "grid_edge_min": float(np.min(densities_a)),
            "grid_edge_var": float(np.var(densities_a)),
            "grid_edge_std": float(np.std(densities_a))
        }
        
        fm_b = {
            "edge_density": density_b,
            "grid_edge_mean": float(np.mean(densities_b)),
            "grid_edge_max": float(np.max(densities_b)),
            "grid_edge_min": float(np.min(densities_b)),
            "grid_edge_var": float(np.var(densities_b)),
            "grid_edge_std": float(np.std(densities_b))
        }
        
        pairwise_res = {
            "whole_edge_density_diff": whole_diff,
            "grid_edge_mean_diff": float(np.mean(cell_diffs)),
            "grid_edge_max_diff": float(np.max(cell_diffs)),
            "grid_edge_min_diff": float(np.min(cell_diffs)),
            "grid_edge_var_diff": float(np.var(cell_diffs)),
            "grid_edge_std_diff": float(np.std(cell_diffs))
        }
        
        elapsed = (time.perf_counter() - t0) * 1000
        logs.append(f"[Edge] Features computed in {elapsed:.1f} ms")
        
        return ExtractorResult(
            frame_a_metrics=fm_a,
            frame_b_metrics=fm_b,
            pairwise_metrics=pairwise_res,
            visuals={"edges_a": edges_a, "edges_b": edges_b, "edge_grid_scores": cell_diffs.tolist()}
        )

class SSIMExtractor(BaseExtractor):
    def extract(self, img_a: np.ndarray, img_b: np.ndarray, config: PairwiseFeatureConfig, cache: dict, logs: list) -> ExtractorResult:
        t0 = time.perf_counter()
        
        hash_a = hash(img_a.tobytes())
        hash_b = hash(img_b.tobytes())
        
        ssim_key = ("ssim", hash_a, hash_b, config.ssim_win_size, config.ssim_gaussian)
        
        # Calculate MAD (always computed, not cached separately but lightweight)
        mad = float(np.mean(np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))))
        
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        
        if ssim_key in cache:
            logs.append(f"[SSIM] Structural similarity cache HIT")
            score, ssim_map = cache[ssim_key]
        else:
            win = config.ssim_win_size
            min_dim = min(gray_a.shape[0], gray_a.shape[1])
            if min_dim < win:
                win = min_dim - (1 if min_dim % 2 == 0 else 0)
                win = max(3, win)
                
            score, ssim_map = structural_similarity(
                gray_a, gray_b,
                win_size=win,
                gaussian_weights=config.ssim_gaussian,
                full=True
            )
            cache[ssim_key] = (score, ssim_map)
            
        pairwise_res = {
            "ssim_mean": float(score),
            "ssim_min": float(np.min(ssim_map)),
            "ssim_variance": float(np.var(ssim_map)),
            "mean_absolute_difference": mad
        }
        
        elapsed = (time.perf_counter() - t0) * 1000
        logs.append(f"[SSIM] Features computed in {elapsed:.1f} ms")
        
        return ExtractorResult(
            frame_a_metrics={},
            frame_b_metrics={},
            pairwise_metrics=pairwise_res,
            visuals={"ssim_map": ssim_map}
        )

class MorphologyExtractor(BaseExtractor):
    def get_text_occupancy_mask(self, img, threshold, kernel_size, iterations, min_area):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary, kernel, iterations=iterations)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
        mask = np.zeros_like(dilated)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        return mask

    def extract(self, img_a: np.ndarray, img_b: np.ndarray, config: PairwiseFeatureConfig, cache: dict, logs: list) -> ExtractorResult:
        t0 = time.perf_counter()
        
        hash_a = hash(img_a.tobytes())
        hash_b = hash(img_b.tobytes())
        
        morph_key_a = ("morph_mask", hash_a, config.text_thresh, config.text_kernel, config.text_iterations, config.text_min_area)
        # Morphology text masks caching
        if morph_key_a in cache:
            logs.append(f"[Morphology] Frame A mask cache HIT")
            mask_a = cache[morph_key_a]
        else:
            mask_a = self.get_text_occupancy_mask(img_a, config.text_thresh, config.text_kernel, config.text_iterations, config.text_min_area)
            cache[morph_key_a] = mask_a
            
        morph_key_b = ("morph_mask", hash_b, config.text_thresh, config.text_kernel, config.text_iterations, config.text_min_area)
        if morph_key_b in cache:
            logs.append(f"[Morphology] Frame B mask cache HIT")
            mask_b = cache[morph_key_b]
        else:
            mask_b = self.get_text_occupancy_mask(img_b, config.text_thresh, config.text_kernel, config.text_iterations, config.text_min_area)
            cache[morph_key_b] = mask_b
            
        occ_a = float(np.sum(mask_a == 255) / mask_a.size)
        occ_b = float(np.sum(mask_b == 255) / mask_b.size)
        
        whole_diff = abs(occ_a - occ_b)
        
        fm_a = {"text_occupancy": occ_a}
        fm_b = {"text_occupancy": occ_b}
        
        pairwise_res = {
            "text_occupancy_diff": whole_diff
        }
        
        elapsed = (time.perf_counter() - t0) * 1000
        logs.append(f"[Morphology] Features computed in {elapsed:.1f} ms")
        
        return ExtractorResult(
            frame_a_metrics=fm_a,
            frame_b_metrics=fm_b,
            pairwise_metrics=pairwise_res,
            visuals={"text_mask_a": mask_a, "text_mask_b": mask_b}
        )

# ==========================================
# 4. Coordinator Engine (Stable Public API)
# ==========================================

class PairwiseFeatureExtractor:
    def __init__(self, config: PairwiseFeatureConfig):
        self.config = config
        self.extractors = []
        
    def register_extractor(self, extractor: BaseExtractor):
        self.extractors.append(extractor)
        
    def extract(self, img_a: np.ndarray, img_b: np.ndarray) -> tuple[FrameFeatures, FrameFeatures, PairwiseFeatures, VisualArtifacts, list[str]]:
        t_start = time.perf_counter()
        
        # Initialize custom cache inside st.session_state if not present
        if "pairwise_cache" not in st.session_state:
            st.session_state["pairwise_cache"] = {}
        cache = st.session_state["pairwise_cache"]
        
        logs = []
        logs.append(f"[System] Starting pairwise feature extraction pipeline...")
        
        # 1. Dimensional Alignment: Resize Frame B to match Frame A
        h_a, w_a = img_a.shape[:2]
        img_b_aligned = cv2.resize(img_b, (w_a, h_a))
        logs.append(f"[System] Dimensions aligned. Processing resolution: {w_a}x{h_a}")
        
        frame_a_dict = {}
        frame_b_dict = {}
        pairwise_dict = {}
        visuals_dict = {}
        
        # 2. Run all registered extractors
        for ext in self.extractors:
            try:
                res = ext.extract(img_a, img_b_aligned, self.config, cache, logs)
                frame_a_dict.update(res.frame_a_metrics)
                frame_b_dict.update(res.frame_b_metrics)
                pairwise_dict.update(res.pairwise_metrics)
                visuals_dict.update(res.visuals)
            except Exception as ex:
                logs.append(f"[Error] Extractor {ext.__class__.__name__} failed: {str(ex)}")
                
        # 3. Create absolute pixel differences for VisualArtifacts
        abs_diff = cv2.absdiff(img_a, img_b_aligned)
        abs_diff_gray = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)
        
        # Binarize difference to compute changed pixel statistics
        _, diff_mask = cv2.threshold(abs_diff_gray, 15, 255, cv2.THRESH_BINARY)
        changed_pixels = int(np.sum(diff_mask == 255))
        changed_pct = float((changed_pixels / diff_mask.size) * 100.0)
        
        visuals_dict["difference_map"] = abs_diff_gray
        visuals_dict["changed_pixel_count"] = changed_pixels
        visuals_dict["changed_pixel_pct"] = changed_pct
        
        # Compile Frame A & B typed models
        fa = FrameFeatures(
            brightness=frame_a_dict.get("brightness", 0.0),
            contrast=frame_a_dict.get("contrast", 0.0),
            entropy=frame_a_dict.get("entropy", 0.0),
            edge_density=frame_a_dict.get("edge_density", 0.0),
            text_occupancy=frame_a_dict.get("text_occupancy", 0.0),
            
            global_rgb_hist_mean=frame_a_dict.get("global_rgb_hist_mean", 0.0),
            global_rgb_hist_max=frame_a_dict.get("global_rgb_hist_max", 0.0),
            global_rgb_hist_min=frame_a_dict.get("global_rgb_hist_min", 0.0),
            global_rgb_hist_var=frame_a_dict.get("global_rgb_hist_var", 0.0),
            global_rgb_hist_std=frame_a_dict.get("global_rgb_hist_std", 0.0),
            
            global_gray_hist_mean=frame_a_dict.get("global_gray_hist_mean", 0.0),
            global_gray_hist_max=frame_a_dict.get("global_gray_hist_max", 0.0),
            global_gray_hist_min=frame_a_dict.get("global_gray_hist_min", 0.0),
            global_gray_hist_var=frame_a_dict.get("global_gray_hist_var", 0.0),
            global_gray_hist_std=frame_a_dict.get("global_gray_hist_std", 0.0),
            
            grid_rgb_hist_mean=frame_a_dict.get("grid_rgb_hist_mean", 0.0),
            grid_rgb_hist_max=frame_a_dict.get("grid_rgb_hist_max", 0.0),
            grid_rgb_hist_min=frame_a_dict.get("grid_rgb_hist_min", 0.0),
            grid_rgb_hist_var=frame_a_dict.get("grid_rgb_hist_var", 0.0),
            grid_rgb_hist_std=frame_a_dict.get("grid_rgb_hist_std", 0.0),
            
            grid_gray_hist_mean=frame_a_dict.get("grid_gray_hist_mean", 0.0),
            grid_gray_hist_max=frame_a_dict.get("grid_gray_hist_max", 0.0),
            grid_gray_hist_min=frame_a_dict.get("grid_gray_hist_min", 0.0),
            grid_gray_hist_var=frame_a_dict.get("grid_gray_hist_var", 0.0),
            grid_gray_hist_std=frame_a_dict.get("grid_gray_hist_std", 0.0),
            
            grid_edge_mean=frame_a_dict.get("grid_edge_mean", 0.0),
            grid_edge_max=frame_a_dict.get("grid_edge_max", 0.0),
            grid_edge_min=frame_a_dict.get("grid_edge_min", 0.0),
            grid_edge_var=frame_a_dict.get("grid_edge_var", 0.0),
            grid_edge_std=frame_a_dict.get("grid_edge_std", 0.0)
        )
        
        fb = FrameFeatures(
            brightness=frame_b_dict.get("brightness", 0.0),
            contrast=frame_b_dict.get("contrast", 0.0),
            entropy=frame_b_dict.get("entropy", 0.0),
            edge_density=frame_b_dict.get("edge_density", 0.0),
            text_occupancy=frame_b_dict.get("text_occupancy", 0.0),
            
            global_rgb_hist_mean=frame_b_dict.get("global_rgb_hist_mean", 0.0),
            global_rgb_hist_max=frame_b_dict.get("global_rgb_hist_max", 0.0),
            global_rgb_hist_min=frame_b_dict.get("global_rgb_hist_min", 0.0),
            global_rgb_hist_var=frame_b_dict.get("global_rgb_hist_var", 0.0),
            global_rgb_hist_std=frame_b_dict.get("global_rgb_hist_std", 0.0),
            
            global_gray_hist_mean=frame_b_dict.get("global_gray_hist_mean", 0.0),
            global_gray_hist_max=frame_b_dict.get("global_gray_hist_max", 0.0),
            global_gray_hist_min=frame_b_dict.get("global_gray_hist_min", 0.0),
            global_gray_hist_var=frame_b_dict.get("global_gray_hist_var", 0.0),
            global_gray_hist_std=frame_b_dict.get("global_gray_hist_std", 0.0),
            
            grid_rgb_hist_mean=frame_b_dict.get("grid_rgb_hist_mean", 0.0),
            grid_rgb_hist_max=frame_b_dict.get("grid_rgb_hist_max", 0.0),
            grid_rgb_hist_min=frame_b_dict.get("grid_rgb_hist_min", 0.0),
            grid_rgb_hist_var=frame_b_dict.get("grid_rgb_hist_var", 0.0),
            grid_rgb_hist_std=frame_b_dict.get("grid_rgb_hist_std", 0.0),
            
            grid_gray_hist_mean=frame_b_dict.get("grid_gray_hist_mean", 0.0),
            grid_gray_hist_max=frame_b_dict.get("grid_gray_hist_max", 0.0),
            grid_gray_hist_min=frame_b_dict.get("grid_gray_hist_min", 0.0),
            grid_gray_hist_var=frame_b_dict.get("grid_gray_hist_var", 0.0),
            grid_gray_hist_std=frame_b_dict.get("grid_gray_hist_std", 0.0),
            
            grid_edge_mean=frame_b_dict.get("grid_edge_mean", 0.0),
            grid_edge_max=frame_b_dict.get("grid_edge_max", 0.0),
            grid_edge_min=frame_b_dict.get("grid_edge_min", 0.0),
            grid_edge_var=frame_b_dict.get("grid_edge_var", 0.0),
            grid_edge_std=frame_b_dict.get("grid_edge_std", 0.0)
        )
        
        # Compile Pairwise typed models
        pf = PairwiseFeatures(
            # Global RGB
            rgb_hist_dist_global_correlation=pairwise_dict.get("rgb_hist_dist_global_correlation", 0.0),
            rgb_hist_dist_global_intersection=pairwise_dict.get("rgb_hist_dist_global_intersection", 0.0),
            rgb_hist_dist_global_bhattacharyya=pairwise_dict.get("rgb_hist_dist_global_bhattacharyya", 0.0),
            rgb_hist_dist_global_chisquare=pairwise_dict.get("rgb_hist_dist_global_chisquare", 0.0),

            # Global Grayscale
            gray_hist_dist_global_correlation=pairwise_dict.get("gray_hist_dist_global_correlation", 0.0),
            gray_hist_dist_global_intersection=pairwise_dict.get("gray_hist_dist_global_intersection", 0.0),
            gray_hist_dist_global_bhattacharyya=pairwise_dict.get("gray_hist_dist_global_bhattacharyya", 0.0),
            gray_hist_dist_global_chisquare=pairwise_dict.get("gray_hist_dist_global_chisquare", 0.0),

            # Grid RGB - Correlation
            rgb_hist_grid_mean_correlation=pairwise_dict.get("rgb_hist_grid_mean_correlation", 0.0),
            rgb_hist_grid_max_correlation=pairwise_dict.get("rgb_hist_grid_max_correlation", 0.0),
            rgb_hist_grid_min_correlation=pairwise_dict.get("rgb_hist_grid_min_correlation", 0.0),
            rgb_hist_grid_var_correlation=pairwise_dict.get("rgb_hist_grid_var_correlation", 0.0),
            rgb_hist_grid_std_correlation=pairwise_dict.get("rgb_hist_grid_std_correlation", 0.0),

            # Grid RGB - Intersection
            rgb_hist_grid_mean_intersection=pairwise_dict.get("rgb_hist_grid_mean_intersection", 0.0),
            rgb_hist_grid_max_intersection=pairwise_dict.get("rgb_hist_grid_max_intersection", 0.0),
            rgb_hist_grid_min_intersection=pairwise_dict.get("rgb_hist_grid_min_intersection", 0.0),
            rgb_hist_grid_var_intersection=pairwise_dict.get("rgb_hist_grid_var_intersection", 0.0),
            rgb_hist_grid_std_intersection=pairwise_dict.get("rgb_hist_grid_std_intersection", 0.0),

            # Grid RGB - Bhattacharyya
            rgb_hist_grid_mean_bhattacharyya=pairwise_dict.get("rgb_hist_grid_mean_bhattacharyya", 0.0),
            rgb_hist_grid_max_bhattacharyya=pairwise_dict.get("rgb_hist_grid_max_bhattacharyya", 0.0),
            rgb_hist_grid_min_bhattacharyya=pairwise_dict.get("rgb_hist_grid_min_bhattacharyya", 0.0),
            rgb_hist_grid_var_bhattacharyya=pairwise_dict.get("rgb_hist_grid_var_bhattacharyya", 0.0),
            rgb_hist_grid_std_bhattacharyya=pairwise_dict.get("rgb_hist_grid_std_bhattacharyya", 0.0),

            # Grid RGB - ChiSquare
            rgb_hist_grid_mean_chisquare=pairwise_dict.get("rgb_hist_grid_mean_chisquare", 0.0),
            rgb_hist_grid_max_chisquare=pairwise_dict.get("rgb_hist_grid_max_chisquare", 0.0),
            rgb_hist_grid_min_chisquare=pairwise_dict.get("rgb_hist_grid_min_chisquare", 0.0),
            rgb_hist_grid_var_chisquare=pairwise_dict.get("rgb_hist_grid_var_chisquare", 0.0),
            rgb_hist_grid_std_chisquare=pairwise_dict.get("rgb_hist_grid_std_chisquare", 0.0),

            # Grid Grayscale - Correlation
            gray_hist_grid_mean_correlation=pairwise_dict.get("gray_hist_grid_mean_correlation", 0.0),
            gray_hist_grid_max_correlation=pairwise_dict.get("gray_hist_grid_max_correlation", 0.0),
            gray_hist_grid_min_correlation=pairwise_dict.get("gray_hist_grid_min_correlation", 0.0),
            gray_hist_grid_var_correlation=pairwise_dict.get("gray_hist_grid_var_correlation", 0.0),
            gray_hist_grid_std_correlation=pairwise_dict.get("gray_hist_grid_std_correlation", 0.0),

            # Grid Grayscale - Intersection
            gray_hist_grid_mean_intersection=pairwise_dict.get("gray_hist_grid_mean_intersection", 0.0),
            gray_hist_grid_max_intersection=pairwise_dict.get("gray_hist_grid_max_intersection", 0.0),
            gray_hist_grid_min_intersection=pairwise_dict.get("gray_hist_grid_min_intersection", 0.0),
            gray_hist_grid_var_intersection=pairwise_dict.get("gray_hist_grid_var_intersection", 0.0),
            gray_hist_grid_std_intersection=pairwise_dict.get("gray_hist_grid_std_intersection", 0.0),

            # Grid Grayscale - Bhattacharyya
            gray_hist_grid_mean_bhattacharyya=pairwise_dict.get("gray_hist_grid_mean_bhattacharyya", 0.0),
            gray_hist_grid_max_bhattacharyya=pairwise_dict.get("gray_hist_grid_max_bhattacharyya", 0.0),
            gray_hist_grid_min_bhattacharyya=pairwise_dict.get("gray_hist_grid_min_bhattacharyya", 0.0),
            gray_hist_grid_var_bhattacharyya=pairwise_dict.get("gray_hist_grid_var_bhattacharyya", 0.0),
            gray_hist_grid_std_bhattacharyya=pairwise_dict.get("gray_hist_grid_std_bhattacharyya", 0.0),

            # Grid Grayscale - ChiSquare
            gray_hist_grid_mean_chisquare=pairwise_dict.get("gray_hist_grid_mean_chisquare", 0.0),
            gray_hist_grid_max_chisquare=pairwise_dict.get("gray_hist_grid_max_chisquare", 0.0),
            gray_hist_grid_min_chisquare=pairwise_dict.get("gray_hist_grid_min_chisquare", 0.0),
            gray_hist_grid_var_chisquare=pairwise_dict.get("gray_hist_grid_var_chisquare", 0.0),
            gray_hist_grid_std_chisquare=pairwise_dict.get("gray_hist_grid_std_chisquare", 0.0),

            # Non-histogram
            whole_edge_density_diff=pairwise_dict.get("whole_edge_density_diff", 0.0),
            grid_edge_mean_diff=pairwise_dict.get("grid_edge_mean_diff", 0.0),
            grid_edge_max_diff=pairwise_dict.get("grid_edge_max_diff", 0.0),
            grid_edge_min_diff=pairwise_dict.get("grid_edge_min_diff", 0.0),
            grid_edge_var_diff=pairwise_dict.get("grid_edge_var_diff", 0.0),
            grid_edge_std_diff=pairwise_dict.get("grid_edge_std_diff", 0.0),
            ssim_mean=pairwise_dict.get("ssim_mean", 0.0),
            ssim_min=pairwise_dict.get("ssim_min", 0.0),
            ssim_variance=pairwise_dict.get("ssim_variance", 0.0),
            mean_absolute_difference=pairwise_dict.get("mean_absolute_difference", 0.0),
            text_occupancy_diff=pairwise_dict.get("text_occupancy_diff", 0.0)
        )
        
        # Compile Visual Artifacts
        artifacts = VisualArtifacts(
            edges_a=visuals_dict.get("edges_a", np.zeros_like(abs_diff_gray)),
            edges_b=visuals_dict.get("edges_b", np.zeros_like(abs_diff_gray)),
            text_mask_a=visuals_dict.get("text_mask_a", np.zeros_like(abs_diff_gray)),
            text_mask_b=visuals_dict.get("text_mask_b", np.zeros_like(abs_diff_gray)),
            ssim_map=visuals_dict.get("ssim_map", np.zeros_like(abs_diff_gray, dtype=np.float32)),
            difference_map=abs_diff_gray,
            changed_pixel_count=changed_pixels,
            changed_pixel_pct=changed_pct,
            hist_grid_scores=visuals_dict.get("hist_grid_scores", []),
            edge_grid_scores=visuals_dict.get("edge_grid_scores", [])
        )
        
        # 4. Run Validator
        FeatureValidator.validate(fa, fb, pf, logs)
        
        total_time = (time.perf_counter() - t_start) * 1000
        logs.append(f"[System] Pipeline complete. Total extraction latency: {total_time:.1f} ms")
        
        return fa, fb, pf, artifacts, logs

# ==========================================
# 5. Validation Framework
# ==========================================

class FeatureValidator:
    @staticmethod
    def validate(fa: FrameFeatures, fb: FrameFeatures, pf: PairwiseFeatures, logs: list):
        # Assert limits
        if not (-1.01 <= pf.ssim_mean <= 1.01):
            logs.append(f"[Warning] Validator check failed: SSIM mean {pf.ssim_mean} is out of bounds [-1, 1]")
        if not (-1.01 <= pf.rgb_hist_dist_global_correlation <= 1.01):
            logs.append(f"[Warning] Validator check failed: Global Correlation score {pf.rgb_hist_dist_global_correlation} is out of bounds")
        if fa.edge_density < 0.0 or fb.edge_density < 0.0:
            logs.append(f"[Warning] Validator check failed: Negative edge density found")
        if fa.text_occupancy < 0.0 or fb.text_occupancy < 0.0:
            logs.append(f"[Warning] Validator check failed: Negative text occupancy found")

# ==========================================
# 6. Exporters
# ==========================================

class CSVExporter:
    @staticmethod
    def export(fa: FrameFeatures, fb: FrameFeatures, pf: PairwiseFeatures) -> str:
        # Create CSV layout: FrameA + FrameB + Pairwise Differences
        all_values = fa.to_list() + fb.to_list() + pf.to_list()
        
        # Serialize list as a single comma-separated row
        return ",".join(str(val) for val in all_values)

# ==========================================
# 8. Decoupled Visualizers
# ==========================================

def draw_grid_overlay(img, grid_size):
    overlay = img.copy()
    h, w = overlay.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size
    for j in range(1, grid_size):
        x = int(j * cell_w)
        cv2.line(overlay, (x, 0), (x, h), (239, 68, 68), 2)
    for i in range(1, grid_size):
        y = int(i * cell_h)
        cv2.line(overlay, (0, y), (w, y), (239, 68, 68), 2)
    return overlay

def visualize_edge_overlay(img, edges):
    overlay = img.copy()
    overlay[edges == 255] = [16, 185, 129]  # Render green edges
    return overlay

def visualize_grid_heatmap(img_shape, cells_diff, grid_size):
    """Generates an NxN visual heatmap representing grid difference scores."""
    h, w = img_shape[:2]
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    max_d = max(cells_diff) if cells_diff else 1.0
    if max_d == 0:
        max_d = 1.0
        
    for idx, diff in enumerate(cells_diff):
        row = idx // grid_size
        col = idx % grid_size
        
        y1, y2 = int(row * cell_h), int(min((row + 1) * cell_h, h))
        x1, x2 = int(col * cell_w), int(min((col + 1) * cell_w, w))
        
        # Color mapping: Green (low diff) -> Yellow (medium) -> Red (high)
        ratio = diff / max_d
        if ratio < 0.5:
            # Green to Yellow
            r = int(ratio * 2 * 255)
            g = 255
            b = 0
        else:
            # Yellow to Red
            r = 255
            g = int((1.0 - ratio) * 2 * 255)
            b = 0
            
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (r, g, b), -1)
        
        # Render boundary lines
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (46, 46, 79), 2)
        
        # Write score
        text = f"{diff:.4f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, cell_w / 250.0)
        thickness = 1
        
        # Place score text in cell center
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + int((cell_w - text_size[0]) / 2)
        text_y = y1 + int((cell_h + text_size[1]) / 2)
        cv2.putText(heatmap, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return heatmap

# ==========================================
# 9. Main Laboratory Tab Coordinator UI
# ==========================================

def render_pairwise_feature_lab():
    st.markdown("### Pairwise Feature Vector Experimentation Lab")
    st.write("Analyze structural and color distributions between Frame A and Frame B interactively.")
    
    # 1. File Uploaders
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file_a = st.file_uploader("Upload Frame A", type=["jpg", "jpeg", "png"], key="upload_v2_a")
    with col_u2:
        file_b = st.file_uploader("Upload Frame B", type=["jpg", "jpeg", "png"], key="upload_v2_b")
        
    if not file_a or not file_b:
        st.info("Upload Frame A and Frame B to run comparisons.")
        return
        
    # Read files
    img_a_pil = Image.open(file_a)
    img_b_pil = Image.open(file_b)
    
    img_a_orig = np.array(img_a_pil.convert("RGB"))
    img_b_orig = np.array(img_b_pil.convert("RGB"))
    
    # Dynamic Configuration Mapping
    config = PairwiseFeatureConfig(
        hist_bins=st.session_state["hist_bins"],
        hist_method=st.session_state["hist_method"],
        color_mode=st.session_state["color_mode"],
        hist_grid_size=st.session_state["hist_grid_size"],
        edge_blur=st.session_state["edge_blur"],
        canny_low=st.session_state["canny_low"],
        canny_high=st.session_state["canny_high"],
        edge_grid_size=st.session_state["edge_grid_size"],
        ssim_win_size=st.session_state["ssim_win_size"],
        ssim_gaussian=st.session_state["ssim_gaussian"],
        text_thresh=st.session_state["text_thresh"],
        text_kernel=st.session_state["text_kernel"],
        text_iterations=st.session_state["text_iterations"],
        text_min_area=st.session_state["text_min_area"],
        hist_epsilon=st.session_state.get("hist_epsilon", 1e-10)
    )
    
    # --- Persistent Experiment Configuration Banner ---
    st.markdown(f"""
    <div style="background-color: #1e1e2f; padding: 1rem; border-radius: 8px; border: 1px solid #2e2e4f; margin-bottom: 1.5rem;">
        <span style="color: #a78bfa; font-weight: bold;">Experiment Config:</span> 
        Bins: <code>{config.hist_bins}</code> | 
        Hist Metric: <code>{config.hist_method}</code> | 
        Grid: <code>{config.edge_grid_size}x{config.edge_grid_size}</code> | 
        Canny: <code>{config.canny_low}/{config.canny_high}</code> | 
        SSIM Window: <code>{config.ssim_win_size}</code>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. RUN PIPELINE (Unified Stable API)
    extractor = PairwiseFeatureExtractor(config)
    extractor.register_extractor(HistogramExtractor())
    extractor.register_extractor(EdgeExtractor())
    extractor.register_extractor(SSIMExtractor())
    extractor.register_extractor(MorphologyExtractor())
    
    fa, fb, pf, art, logs = extractor.extract(img_a_orig, img_b_orig)
    
    # --- VISUALIZATIONS ---
    st.markdown("---")
    st.markdown("### 📊 Visualizations Panel")
    
    # 1. Overlay grids
    tab_grids, tab_edges, tab_text, tab_diffs = st.tabs(["Original / Grids", "Canny Edge Overlays", "Morphology Text Masks", "Pixel / SSIM Diffs"])
    
    with tab_grids:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(draw_grid_overlay(img_a_orig, config.edge_grid_size), caption="Frame A Grid Overlay", use_container_width=True)
        with col_g2:
            h_a, w_a = img_a_orig.shape[:2]
            img_b_resized = cv2.resize(img_b_orig, (w_a, h_a))
            st.image(draw_grid_overlay(img_b_resized, config.edge_grid_size), caption="Frame B Grid Overlay (Resized)", use_container_width=True)
            
    with tab_edges:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.image(visualize_edge_overlay(img_a_orig, art.edges_a), caption="Frame A Edge Overlay", use_container_width=True)
        with col_e2:
            st.image(visualize_edge_overlay(img_b_resized, art.edges_b), caption="Frame B Edge Overlay", use_container_width=True)
            
    with tab_text:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image(art.text_mask_a, caption="Frame A Text Mask", use_container_width=True)
        with col_m2:
            st.image(art.text_mask_b, caption="Frame B Text Mask", use_container_width=True)
            
    with tab_diffs:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.image(art.difference_map, caption=f"Pixel Difference (Changed: {art.changed_pixel_count} px / {art.changed_pixel_pct:.2f}%)", use_container_width=True)
        with col_d2:
            # SSIM maps normalization
            ssim_vis = ((art.ssim_map + 1.0) * 127.5).astype(np.uint8)
            st.image(ssim_vis, caption="SSIM Similarity Map (Bright = High Similarity)", use_container_width=True)
            
    # Spatial difference heatmap (No comparative horizontal bar plot, as requested)
    st.markdown("#### Spatial Difference Heatmap")
    heatmap = visualize_grid_heatmap(img_a_orig.shape, art.edge_grid_scores, config.edge_grid_size)
    st.image(heatmap, caption="Cell Edge Density Grid Differences Heatmap", use_container_width=True)
        
    # --- FEATURES TABLE ---
    st.markdown("---")
    st.markdown("### 📋 Computed Features Tables")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Individual Frame Scalar & Raw Features (Section A)**")
        df_frame = pd.DataFrame({
            "Feature Metric": [
                "Brightness (Mean)", "Contrast (Std)", "Shannon Entropy", "Edge Density", "Text Occupancy",
                "Global RGB Histogram (Mean / Max / Min / Var / Std)",
                "Global Gray Histogram (Mean / Max / Min / Var / Std)",
                "Grid RGB Histogram (Mean / Max / Min / Var / Std)",
                "Grid Gray Histogram (Mean / Max / Min / Var / Std)",
                "Grid Edge Density (Mean / Max / Min / Var / Std)"
            ],
            "Frame A": [
                f"{fa.brightness:.4f}", f"{fa.contrast:.4f}", f"{fa.entropy:.4f}", f"{fa.edge_density:.4f}", f"{fa.text_occupancy:.4f}",
                f"{fa.global_rgb_hist_mean:.4f} / {fa.global_rgb_hist_max:.4f} / {fa.global_rgb_hist_min:.4f} / {fa.global_rgb_hist_var:.4f} / {fa.global_rgb_hist_std:.4f}",
                f"{fa.global_gray_hist_mean:.4f} / {fa.global_gray_hist_max:.4f} / {fa.global_gray_hist_min:.4f} / {fa.global_gray_hist_var:.4f} / {fa.global_gray_hist_std:.4f}",
                f"{fa.grid_rgb_hist_mean:.4f} / {fa.grid_rgb_hist_max:.4f} / {fa.grid_rgb_hist_min:.4f} / {fa.grid_rgb_hist_var:.4f} / {fa.grid_rgb_hist_std:.4f}",
                f"{fa.grid_gray_hist_mean:.4f} / {fa.grid_gray_hist_max:.4f} / {fa.grid_gray_hist_min:.4f} / {fa.grid_gray_hist_var:.4f} / {fa.grid_gray_hist_std:.4f}",
                f"{fa.grid_edge_mean:.4f} / {fa.grid_edge_max:.4f} / {fa.grid_edge_min:.4f} / {fa.grid_edge_var:.4f} / {fa.grid_edge_std:.4f}"
            ],
            "Frame B": [
                f"{fb.brightness:.4f}", f"{fb.contrast:.4f}", f"{fb.entropy:.4f}", f"{fb.edge_density:.4f}", f"{fb.text_occupancy:.4f}",
                f"{fb.global_rgb_hist_mean:.4f} / {fb.global_rgb_hist_max:.4f} / {fb.global_rgb_hist_min:.4f} / {fb.global_rgb_hist_var:.4f} / {fb.global_rgb_hist_std:.4f}",
                f"{fb.global_gray_hist_mean:.4f} / {fb.global_gray_hist_max:.4f} / {fb.global_gray_hist_min:.4f} / {fb.global_gray_hist_var:.4f} / {fb.global_gray_hist_std:.4f}",
                f"{fb.grid_rgb_hist_mean:.4f} / {fb.grid_rgb_hist_max:.4f} / {fb.grid_rgb_hist_min:.4f} / {fb.grid_rgb_hist_var:.4f} / {fb.grid_rgb_hist_std:.4f}",
                f"{fb.grid_gray_hist_mean:.4f} / {fb.grid_gray_hist_max:.4f} / {fb.grid_gray_hist_min:.4f} / {fb.grid_gray_hist_var:.4f} / {fb.grid_gray_hist_std:.4f}",
                f"{fb.grid_edge_mean:.4f} / {fb.grid_edge_max:.4f} / {fb.grid_edge_min:.4f} / {fb.grid_edge_var:.4f} / {fb.grid_edge_std:.4f}"
            ]
        })
        st.dataframe(df_frame, use_container_width=True, hide_index=True)
        
    with col_t2:
        st.markdown("**Comparative Pairwise Metric Statistics (Section B)**")
        df_pairwise = pd.DataFrame({
            "Pairwise Feature Metric": [
                "Global RGB Correlation (Higher = Better)", "Global RGB Intersection (Higher = Better)", "Global RGB Bhattacharyya (Lower = Better)", "Global RGB Chi-Square (Lower = Better)",
                "Global Gray Correlation (Higher = Better)", "Global Gray Intersection (Higher = Better)", "Global Gray Bhattacharyya (Lower = Better)", "Global Gray Chi-Square (Lower = Better)",
                "Grid RGB Correlation (Mean / Max / Min / Var / Std)",
                "Grid RGB Intersection (Mean / Max / Min / Var / Std)",
                "Grid RGB Bhattacharyya (Mean / Max / Min / Var / Std)",
                "Grid RGB Chi-Square (Mean / Max / Min / Var / Std)",
                "Grid Gray Correlation (Mean / Max / Min / Var / Std)",
                "Grid Gray Intersection (Mean / Max / Min / Var / Std)",
                "Grid Gray Bhattacharyya (Mean / Max / Min / Var / Std)",
                "Grid Gray Chi-Square (Mean / Max / Min / Var / Std)",
                "Whole Edge Density Diff",
                "Grid Edge Difference (Mean / Max / Min / Var / Std)",
                "SSIM Map Stats (Mean / Min / Var)",
                "Mean Absolute Difference (MAD)",
                "Text Occupancy Diff"
            ],
            "Pairwise Metric Value": [
                f"{pf.rgb_hist_dist_global_correlation:.4f}", f"{pf.rgb_hist_dist_global_intersection:.4f}", f"{pf.rgb_hist_dist_global_bhattacharyya:.4f}", f"{pf.rgb_hist_dist_global_chisquare:.4f}",
                f"{pf.gray_hist_dist_global_correlation:.4f}", f"{pf.gray_hist_dist_global_intersection:.4f}", f"{pf.gray_hist_dist_global_bhattacharyya:.4f}", f"{pf.gray_hist_dist_global_chisquare:.4f}",
                f"{pf.rgb_hist_grid_mean_correlation:.4f} / {pf.rgb_hist_grid_max_correlation:.4f} / {pf.rgb_hist_grid_min_correlation:.4f} / {pf.rgb_hist_grid_var_correlation:.4f} / {pf.rgb_hist_grid_std_correlation:.4f}",
                f"{pf.rgb_hist_grid_mean_intersection:.4f} / {pf.rgb_hist_grid_max_intersection:.4f} / {pf.rgb_hist_grid_min_intersection:.4f} / {pf.rgb_hist_grid_var_intersection:.4f} / {pf.rgb_hist_grid_std_intersection:.4f}",
                f"{pf.rgb_hist_grid_mean_bhattacharyya:.4f} / {pf.rgb_hist_grid_max_bhattacharyya:.4f} / {pf.rgb_hist_grid_min_bhattacharyya:.4f} / {pf.rgb_hist_grid_var_bhattacharyya:.4f} / {pf.rgb_hist_grid_std_bhattacharyya:.4f}",
                f"{pf.rgb_hist_grid_mean_chisquare:.4f} / {pf.rgb_hist_grid_max_chisquare:.4f} / {pf.rgb_hist_grid_min_chisquare:.4f} / {pf.rgb_hist_grid_var_chisquare:.4f} / {pf.rgb_hist_grid_std_chisquare:.4f}",
                f"{pf.gray_hist_grid_mean_correlation:.4f} / {pf.gray_hist_grid_max_correlation:.4f} / {pf.gray_hist_grid_min_correlation:.4f} / {pf.gray_hist_grid_var_correlation:.4f} / {pf.gray_hist_grid_std_correlation:.4f}",
                f"{pf.gray_hist_grid_mean_intersection:.4f} / {pf.gray_hist_grid_max_intersection:.4f} / {pf.gray_hist_grid_min_intersection:.4f} / {pf.gray_hist_grid_var_intersection:.4f} / {pf.gray_hist_grid_std_intersection:.4f}",
                f"{pf.gray_hist_grid_mean_bhattacharyya:.4f} / {pf.gray_hist_grid_max_bhattacharyya:.4f} / {pf.gray_hist_grid_min_bhattacharyya:.4f} / {pf.gray_hist_grid_var_bhattacharyya:.4f} / {pf.gray_hist_grid_std_bhattacharyya:.4f}",
                f"{pf.gray_hist_grid_mean_chisquare:.4f} / {pf.gray_hist_grid_max_chisquare:.4f} / {pf.gray_hist_grid_min_chisquare:.4f} / {pf.gray_hist_grid_var_chisquare:.4f} / {pf.gray_hist_grid_std_chisquare:.4f}",
                f"{pf.whole_edge_density_diff:.4f}",
                f"{pf.grid_edge_mean_diff:.4f} / {pf.grid_edge_max_diff:.4f} / {pf.grid_edge_min_diff:.4f} / {pf.grid_edge_var_diff:.4f} / {pf.grid_edge_std_diff:.4f}",
                f"{pf.ssim_mean:.4f} / {pf.ssim_min:.4f} / {pf.ssim_variance:.4f}",
                f"{pf.mean_absolute_difference:.4f}",
                f"{pf.text_occupancy_diff:.4f}"
            ]
        })
        st.dataframe(df_pairwise, use_container_width=True, hide_index=True)
        
    # --- REPORT PANEL & EXPORTS ---
    st.markdown("---")
    
    st.markdown("#### 💾 CSV Export Preview (119 Features, Headerless)")
    final_csv_row = CSVExporter.export(fa, fb, pf)
    
    st.code(final_csv_row, language="text")
    
    st.download_button(
        label="💾 Download Pairwise Vector CSV (119 Features, Headerless)",
        data=final_csv_row,
        file_name="pairwise_vector.csv",
        mime="text/csv",
        use_container_width=True
    )
        
    # --- DEBUG CONSOLE LOGS ---
    st.markdown("---")
    with st.expander("🛠️ Debug Console Logs", expanded=False):
        for line in logs:
            st.text(line)
