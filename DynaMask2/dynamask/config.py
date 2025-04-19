import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class DynaMaskConfig:
    """DynaMask2の設定クラス - すべてのチューニング可能なパラメータを整理"""

    # ===== 入力/出力設定 =====
    # 入力設定
    input_type: str = "images"
    input_path: Optional[str] = "/home/rkmtlabkei/data/demo/frames"
    image_pattern: str = "*.jpeg"

    # 出力設定
    output_dir: Optional[str] = None
    output_filename: str = "dynamic_masked.mp4"
    save_debug_frames: bool = True
    debug_dir: Optional[str] = None
    masks_dir: Optional[str] = None
    save_masks: bool = True

    # ===== 動き検出設定 =====
    motion_threshold: float = 80.0
    camera_compensation: bool = True
    optical_flow_winsize: int = 15

    # ===== 自己運動分離設定 (COLMAP) =====
    colmap_images_path: Optional[str] = None
    colmap_cameras_path: Optional[str] = None
    use_colmap_egomotion: bool = True
    fallback_to_flow_compensation: bool = True

    # ===== 領域適応閾値設定 =====
    use_region_adaptive_threshold: bool = True
    hand_region_radius: int = 100
    hand_region_threshold_factor: float = 0.5

    # ===== セグメント判定設定 =====
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.25
    temporal_consistency: int = 3
    motion_overlap_ratio: float = 0.45
    min_motion_pixels: int = 100
    center_motion_ratio: float = 0.35

    # ===== 人間検出設定 =====
    use_pose_detection: bool = True
    pose_confidence: float = 0.5
    hand_confidence: float = 0.5
    hand_proximity_threshold: int = 50
    hand_proximity_factor: float = 0.7
    use_yolo: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.4
    human_overlap_ratio: float = 0.4

    # ===== 内部パラメータ =====
    # FastSAM設定（型ヒントをAnyに変更し、初期化ロジックは外部で行う）
    fastsam_config: Optional[Any] = None
    fastsam_model_name: str = "FastSAM-x"
    fastsam_confidence: float = 0.7
    imgsz: int = 1024

    motion_history_frames: int = 3
    dynamic_decay_rate: float = 0.7
    temporal_threshold_offset: float = 0.8

    def __post_init__(self):
        """初期化後の処理 (出力パス設定のみ)"""
        # 出力ディレクトリの設定
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("output", f"dyna_{timestamp}")

        # デバッグディレクトリの設定
        if self.save_debug_frames and self.debug_dir is None:
            self.debug_dir = os.path.join(self.output_dir, "debug")

        # マスクディレクトリの設定
        if self.save_masks and self.masks_dir is None:
            self.masks_dir = os.path.join(self.output_dir, "masks")
