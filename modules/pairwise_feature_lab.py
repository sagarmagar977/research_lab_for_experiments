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
    rgb_hist_dist_global: float
    gray_hist_dist_global: float
    rgb_hist_grid_mean: float
    rgb_hist_grid_max: float
    rgb_hist_grid_min: float
    rgb_hist_grid_var: float
    rgb_hist_grid_std: float
    gray_hist_grid_mean: float
    gray_hist_grid_max: float
    gray_hist_grid_min: float
    gray_hist_grid_var: float
    gray_hist_grid_std: float
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
            self.rgb_hist_dist_global,
            self.gray_hist_dist_global,
            self.rgb_hist_grid_mean,
            self.rgb_hist_grid_max,
            self.rgb_hist_grid_min,
            self.rgb_hist_grid_var,
            self.rgb_hist_grid_std,
            self.gray_hist_grid_mean,
            self.gray_hist_grid_max,
            self.gray_hist_grid_min,
            self.gray_hist_grid_var,
            self.gray_hist_grid_std,
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

    def calc_norm_hist(self, img, bins, color_mode):
        if color_mode == "Grayscale":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist
        else:
            hists = []
            for ch in range(3):
                hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hists.append(hist)
            return np.concatenate(hists, axis=0)

    def compare_hist(self, hist1, hist2, method_str):
        m_map = {
            "Correlation": cv2.HISTCMP_CORREL,
            "Chi-Square": cv2.HISTCMP_CHISQR,
            "Intersection": cv2.HISTCMP_INTERSECT,
            "Bhattacharyya": cv2.HISTCMP_BHATTACHARYYA
        }
        mid = m_map.get(method_str, cv2.HISTCMP_CORREL)
        return float(cv2.compareHist(hist1, hist2, mid))

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
        hist_rgb_key_a = ("hist_rgb", hash_a, config.hist_bins)
        hist_rgb_key_b = ("hist_rgb", hash_b, config.hist_bins)
        hist_gray_key_a = ("hist_gray", hash_a, config.hist_bins)
        hist_gray_key_b = ("hist_gray", hash_b, config.hist_bins)
        
        if hist_rgb_key_a not in cache:
            cache[hist_rgb_key_a] = self.calc_norm_hist(img_a, config.hist_bins, "RGB")
        if hist_rgb_key_b not in cache:
            cache[hist_rgb_key_b] = self.calc_norm_hist(img_b, config.hist_bins, "RGB")
        if hist_gray_key_a not in cache:
            cache[hist_gray_key_a] = self.calc_norm_hist(img_a, config.hist_bins, "Grayscale")
        if hist_gray_key_b not in cache:
            cache[hist_gray_key_b] = self.calc_norm_hist(img_b, config.hist_bins, "Grayscale")
            
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
        
        rgb_global_dist = self.compare_hist(rgb_hist_a, rgb_hist_b, config.hist_method)
        gray_global_dist = self.compare_hist(gray_hist_a, gray_hist_b, config.hist_method)
        
        # Grid-based histograms
        cells_a = self.get_grid_cells(img_a.shape, config.hist_grid_size)
        cells_b = self.get_grid_cells(img_b.shape, config.hist_grid_size)
        
        rgb_grid_scores = []
        gray_grid_scores = []
        
        grid_rgb_means_a, grid_rgb_maxes_a, grid_rgb_mins_a, grid_rgb_vars_a, grid_rgb_stds_a = [], [], [], [], []
        grid_gray_means_a, grid_gray_maxes_a, grid_gray_mins_a, grid_gray_vars_a, grid_gray_stds_a = [], [], [], [], []
        
        grid_rgb_means_b, grid_rgb_maxes_b, grid_rgb_mins_b, grid_rgb_vars_b, grid_rgb_stds_b = [], [], [], [], []
        grid_gray_means_b, grid_gray_maxes_b, grid_gray_mins_b, grid_gray_vars_b, grid_gray_stds_b = [], [], [], [], []
        
        for idx, ((ya1, ya2, xa1, xa2), (yb1, yb2, xb1, xb2)) in enumerate(zip(cells_a, cells_b)):
            cell_a = img_a[ya1:ya2, xa1:xa2]
            cell_b = img_b[yb1:yb2, xb1:xb2]
            
            c_rgb_a = self.calc_norm_hist(cell_a, config.hist_bins, "RGB")
            c_rgb_b = self.calc_norm_hist(cell_b, config.hist_bins, "RGB")
            c_gray_a = self.calc_norm_hist(cell_a, config.hist_bins, "Grayscale")
            c_gray_b = self.calc_norm_hist(cell_b, config.hist_bins, "Grayscale")
            
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
            
            rgb_grid_scores.append(self.compare_hist(c_rgb_a, c_rgb_b, config.hist_method))
            gray_grid_scores.append(self.compare_hist(c_gray_a, c_gray_b, config.hist_method))
            
        rgb_grid_scores = np.array(rgb_grid_scores)
        gray_grid_scores = np.array(gray_grid_scores)
        
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
            "rgb_hist_dist_global": rgb_global_dist,
            "gray_hist_dist_global": gray_global_dist,
            
            "rgb_hist_grid_mean": float(np.mean(rgb_grid_scores)),
            "rgb_hist_grid_max": float(np.max(rgb_grid_scores)),
            "rgb_hist_grid_min": float(np.min(rgb_grid_scores)),
            "rgb_hist_grid_var": float(np.var(rgb_grid_scores)),
            "rgb_hist_grid_std": float(np.std(rgb_grid_scores)),
            
            "gray_hist_grid_mean": float(np.mean(gray_grid_scores)),
            "gray_hist_grid_max": float(np.max(gray_grid_scores)),
            "gray_hist_grid_min": float(np.min(gray_grid_scores)),
            "gray_hist_grid_var": float(np.var(gray_grid_scores)),
            "gray_hist_grid_std": float(np.std(gray_grid_scores))
        }
        
        elapsed = (time.perf_counter() - t0) * 1000
        logs.append(f"[Histogram] Features computed in {elapsed:.1f} ms")
        
        return ExtractorResult(
            frame_a_metrics=fm_a,
            frame_b_metrics=fm_b,
            pairwise_metrics=pairwise_res,
            visuals={"hist_grid_scores": rgb_grid_scores.tolist()}
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
            rgb_hist_dist_global=pairwise_dict.get("rgb_hist_dist_global", 0.0),
            gray_hist_dist_global=pairwise_dict.get("gray_hist_dist_global", 0.0),
            
            rgb_hist_grid_mean=pairwise_dict.get("rgb_hist_grid_mean", 0.0),
            rgb_hist_grid_max=pairwise_dict.get("rgb_hist_grid_max", 0.0),
            rgb_hist_grid_min=pairwise_dict.get("rgb_hist_grid_min", 0.0),
            rgb_hist_grid_var=pairwise_dict.get("rgb_hist_grid_var", 0.0),
            rgb_hist_grid_std=pairwise_dict.get("rgb_hist_grid_std", 0.0),
            
            gray_hist_grid_mean=pairwise_dict.get("gray_hist_grid_mean", 0.0),
            gray_hist_grid_max=pairwise_dict.get("gray_hist_grid_max", 0.0),
            gray_hist_grid_min=pairwise_dict.get("gray_hist_grid_min", 0.0),
            gray_hist_grid_var=pairwise_dict.get("gray_hist_grid_var", 0.0),
            gray_hist_grid_std=pairwise_dict.get("gray_hist_grid_std", 0.0),
            
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
        if pf.rgb_hist_dist_global < -1.01:
            logs.append(f"[Warning] Validator check failed: Global RGB histogram score {pf.rgb_hist_dist_global} is out of bounds")
        if fa.edge_density < 0.0 or fb.edge_density < 0.0:
            logs.append(f"[Warning] Validator check failed: Negative edge density found")
        if fa.text_occupancy < 0.0 or fb.text_occupancy < 0.0:
            logs.append(f"[Warning] Validator check failed: Negative text occupancy found")

# ==========================================
# 6. Exporters
# ==========================================

class CSVExporter:
    @staticmethod
    def export(frame_a_name: str, frame_b_name: str, fa: FrameFeatures, fb: FrameFeatures, pf: PairwiseFeatures, config: PairwiseFeatureConfig, ground_truth: int = 0) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = [
            frame_a_name,
            frame_b_name,
            "2.0.0",  # Feature Engine Version
            "2.0.0",  # Feature Schema Version
            "1.0.0",  # Experiment Version
            timestamp,
            0,        # Placeholder for width, filled dynamically in render
            0,        # Placeholder for height, filled dynamically in render
            config.edge_grid_size,
            config.hist_bins,
            config.hist_method,
            config.color_mode,
            config.ssim_win_size,
            json.dumps(asdict(config)).replace(",", ";"),  # Replace comma to prevent CSV formatting corruption
            ground_truth
        ]
        
        # Create CSV layout: Metadata + FrameA + FrameB + Pairwise Differences
        all_values = metadata + fa.to_list() + fb.to_list() + pf.to_list()
        
        # Serialize list as a single comma-separated row
        return ",".join(str(val) for val in all_values)

class MarkdownExporter:
    @staticmethod
    def export(fa: FrameFeatures, fb: FrameFeatures, pf: PairwiseFeatures, config: PairwiseFeatureConfig) -> str:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        explanation = rule_based_explain(fa, fb, pf)
        
        report = f"""# Pairwise Feature Vector Experiment Report
**Date Generated:** {timestamp}
**Engine Version:** 2.0.0
**Schema Version:** 2.0.0

---

## 1. Experiment Hyperparameters Configuration
*   **Histogram Bins:** {config.hist_bins}
*   **Histogram Comparison Metric:** {config.hist_method}
*   **Color Mode:** {config.color_mode}
*   **Histogram Grid Size:** {config.hist_grid_size}x{config.hist_grid_size}
*   **Gaussian Blur Pre-filter:** {config.edge_blur}
*   **Canny Thresholds (Low/High):** {config.canny_low} / {config.canny_high}
*   **Edge Grid Size:** {config.edge_grid_size}x{config.edge_grid_size}
*   **SSIM Window Size:** {config.ssim_win_size}
*   **SSIM Gaussian Weights:** {config.ssim_gaussian}

---

## 2. Extraction Results Summary

### Frame Features (Scalar Metrics)
| Metric | Frame A | Frame B |
| :--- | :--- | :--- |
| Brightness | {fa.brightness:.4f} | {fb.brightness:.4f} |
| Contrast | {fa.contrast:.4f} | {fb.contrast:.4f} |
| Shannon Entropy | {fa.entropy:.4f} | {fb.entropy:.4f} |
| Edge Density | {fa.edge_density:.4f} | {fb.edge_density:.4f} |
| Text Occupancy | {fa.text_occupancy:.4f} | {fb.text_occupancy:.4f} |

### Comparative Metrics (Pairwise Differences)
| Metric | Computed Value |
| :--- | :--- |
| Global RGB Histogram Distance | {pf.rgb_hist_dist_global:.4f} |
| Global Gray Histogram Distance | {pf.gray_hist_dist_global:.4f} |
| Grid RGB Hist Distance (Mean) | {pf.rgb_hist_grid_mean:.4f} |
| Grid RGB Hist Distance (Max) | {pf.rgb_hist_grid_max:.4f} |
| Grid RGB Hist Distance (Min) | {pf.rgb_hist_grid_min:.4f} |
| Grid RGB Hist Distance (Var) | {pf.rgb_hist_grid_var:.4f} |
| Grid Gray Hist Distance (Mean) | {pf.gray_hist_grid_mean:.4f} |
| Grid Gray Hist Distance (Max) | {pf.gray_hist_grid_max:.4f} |
| Grid Gray Hist Distance (Min) | {pf.gray_hist_grid_min:.4f} |
| Grid Gray Hist Distance (Var) | {pf.gray_hist_grid_var:.4f} |
| Whole Image Edge Density Diff | {pf.whole_edge_density_diff:.4f} |
| Grid Edge Difference (Mean) | {pf.grid_edge_mean_diff:.4f} |
| Grid Edge Difference (Max) | {pf.grid_edge_max_diff:.4f} |
| Grid Edge Difference (Min) | {pf.grid_edge_min_diff:.4f} |
| Grid Edge Difference (Var) | {pf.grid_edge_var_diff:.4f} |
| SSIM (Mean Similarity) | {pf.ssim_mean:.4f} |
| SSIM (Minimum similarity) | {pf.ssim_min:.4f} |
| SSIM (Variance) | {pf.ssim_variance:.4f} |
| Mean Absolute Difference (MAD) | {pf.mean_absolute_difference:.4f} |
| Text Occupancy Difference | {pf.text_occupancy_diff:.4f} |

---

## 3. Structural Modification Analysis
{explanation}
"""
        return report

# ==========================================
# 7. Rule-Based Interpretation
# ==========================================

def rule_based_explain(fa: FrameFeatures, fb: FrameFeatures, pf: PairwiseFeatures) -> str:
    lines = []
    
    # 1. SSIM structural assessment
    if pf.ssim_mean > 0.98:
        lines.append("*   **SSIM:** Very high structural similarity (layout identical, tiny changes).")
    elif pf.ssim_mean > 0.90:
        lines.append("*   **SSIM:** High structural similarity (same slide template, minor additions).")
    else:
        lines.append("*   **SSIM:** Low similarity. Substantial layout structure updates or slide transition.")

    # 2. Histogram differences
    if pf.gray_hist_dist_global < 0.05:
        lines.append("*   **Histogram:** Very small appearance difference.")
    elif pf.gray_hist_dist_global < 0.15:
        lines.append("*   **Histogram:** Moderate appearance updates.")
    else:
        lines.append("*   **Histogram:** Large appearance shifts (colors or layout elements swapped).")
        
    # 3. Grid Edge modifications
    if pf.grid_edge_max_diff > 0.08:
        lines.append("*   **Edge:** Localized structural change detected (e.g. annotations drawn or bullet points added).")
    else:
        lines.append("*   **Edge:** Uniform edge difference across grids.")
        
    # 4. Text occupancy additions
    if pf.text_occupancy_diff > 0.03:
        lines.append("*   **Text Occupancy:** Text coverage changed. New text blocks likely appeared.")
        
    # 5. Overall Synthesis
    if pf.ssim_mean > 0.95 and pf.text_occupancy_diff > 0.02 and pf.grid_edge_max_diff > 0.05:
        verdict = "**Overall Verdict:** Likely answer reveal or new text insertion on the same slide structure."
    elif pf.ssim_mean < 0.88:
        verdict = "**Overall Verdict:** Slide transition. Completely new layout template detected."
    elif pf.ssim_mean > 0.98 and pf.mean_absolute_difference < 1.0:
        verdict = "**Overall Verdict:** Virtually identical frames (static slide)."
    else:
        verdict = "**Overall Verdict:** Slide modification with moderate annotations or layout adjustments."
        
    return "\n".join(lines) + "\n\n" + verdict

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
        text_min_area=st.session_state["text_min_area"]
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
                "Global RGB Histogram Distance", "Global Gray Histogram Distance",
                "Grid RGB Histogram Dist (Mean / Max / Min / Var / Std)",
                "Grid Gray Histogram Dist (Mean / Max / Min / Var / Std)",
                "Whole Edge Density Diff",
                "Grid Edge Difference (Mean / Max / Min / Var / Std)",
                "SSIM Map Stats (Mean / Min / Var)",
                "Mean Absolute Difference (MAD)",
                "Text Occupancy Diff"
            ],
            "Pairwise Metric Value": [
                f"{pf.rgb_hist_dist_global:.4f}", f"{pf.gray_hist_dist_global:.4f}",
                f"{pf.rgb_hist_grid_mean:.4f} / {pf.rgb_hist_grid_max:.4f} / {pf.rgb_hist_grid_min:.4f} / {pf.rgb_hist_grid_var:.4f} / {pf.rgb_hist_grid_std:.4f}",
                f"{pf.gray_hist_grid_mean:.4f} / {pf.gray_hist_grid_max:.4f} / {pf.gray_hist_grid_min:.4f} / {pf.gray_hist_grid_var:.4f} / {pf.gray_hist_grid_std:.4f}",
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
    
    col_rep, col_csv = st.columns(2)
    with col_rep:
        st.markdown("#### 🔬 Metric Interpretation & Reports")
        st.write(rule_based_explain(fa, fb, pf))
        
        md_text = MarkdownExporter.export(fa, fb, pf, config)
        st.download_button(
            label="📄 Generate Experiment Report (.md)",
            data=md_text,
            file_name=f"experiment_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
    with col_csv:
        st.markdown("#### 💾 CSV Export Preview")
        # Prepare CSV exporter and inject actual frame width/height
        raw_csv_row = CSVExporter.export(file_a.name, file_b.name, fa, fb, pf, config)
        csv_parts = raw_csv_row.split(",")
        # Inject width and height
        h_orig, w_orig = img_a_orig.shape[:2]
        csv_parts[6] = str(w_orig)
        csv_parts[7] = str(h_orig)
        final_csv_row = ",".join(csv_parts)
        
        st.code(final_csv_row, language="text")
        
        st.download_button(
            label="💾 Download Pairwise Vector CSV (Headerless)",
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
