"""
Level 2 Package:
- Level 2.1: Candidate Frame Grouping (GroupingEngine, Feature Extractor, Reporting)
- Level 2.2: Representative Keyframe Selection (KeyframeSelectorL2_2, QualityCache)
"""
from .grouping_engine import GroupingEngine
from .quality_cache import load_or_compute_quality_cache, resolve_candidate_image_dir
from .keyframe_selector import KeyframeSelectorL2_2
