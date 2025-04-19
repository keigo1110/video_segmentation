import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DynaMaskConfig

logger = logging.getLogger(__name__)

def filter_dynamic_segments(segments: Any, motion_mask: np.ndarray, 
                           frame_shape: Tuple[int, int], config: 'DynaMaskConfig',
                           human_mask: Optional[np.ndarray] = None,
                           hand_positions: Optional[List[Tuple[int, int]]] = None) -> List[int]:
    """動的なセグメントのインデックスをフィルタリングする
    
    Args:
        segments: セグメントデータ (通常は Ultralytics の Results オブジェクト)
        motion_mask: 動きマスク (uint8, 0 or 255)
        frame_shape: フレームサイズ (height, width)
        config: 設定オブジェクト
        human_mask: 人間マスク (uint8, 0-255, 信頼度を表す場合がある)
        hand_positions: 手の位置リスト [(x, y), ...]
    
    Returns:
        List[int]: 動的セグメントのインデックスリスト
    """
    dynamic_indices = []
    height, width = frame_shape[:2]
    total_pixels = height * width

    try:
        # セグメントマスクの取得 (Ultralyticsの結果オブジェクトを想定)
        masks = None
        # Check if segments is a list/tuple (e.g., from batch processing) or single result
        if isinstance(segments, (list, tuple)) and len(segments) > 0:
            result_obj = segments[0] # Assume first result is representative
        elif hasattr(segments, 'masks'): # Assume it's a single result object
             result_obj = segments
        else:
            logger.warning("無効なセグメントデータ形式。フィルタリングをスキップします。")
            return []

        if hasattr(result_obj, 'masks') and result_obj.masks is not None:
            # Ensure masks.data is accessible and is a tensor/numpy array
            if hasattr(result_obj.masks, 'data') and result_obj.masks.data is not None:
                masks_tensor = result_obj.masks.data
                if hasattr(masks_tensor, 'cpu'): # Check if it's a torch tensor
                    masks = masks_tensor.cpu().numpy()
                elif isinstance(masks_tensor, np.ndarray):
                    masks = masks_tensor
                else:
                     logger.warning("不明なマスクデータ形式です。")
                     return []
            elif isinstance(result_obj.masks, np.ndarray):
                 masks = result_obj.masks # Assume it's already a numpy array
            else:
                 logger.warning("マスクデータが見つかりません。")
                 return []
        else:
             logger.warning("セグメントオブジェクトにマスク属性が見つかりません。")
             return []

        if masks is None or len(masks) == 0:
            # logger.debug("フィルタリング対象のマスクが見つかりません。") # This might be too verbose
            return []

        # 動きマスクの前処理 (念のためバイナリ化)
        motion_mask_binary = (motion_mask > 127).astype(np.uint8) * 255
        total_motion_pixels = np.sum(motion_mask_binary > 0)

        # 人間マスクの前処理
        human_confidence = 0.0
        human_pixels = 0
        human_mask_binary = None
        if human_mask is not None:
            # Ensure human_mask is uint8
            if human_mask.dtype != np.uint8:
                 human_mask = human_mask.astype(np.uint8)
            human_mask_binary = (human_mask > 127).astype(np.uint8) # Use a fixed threshold for binary operations
            human_pixels = np.sum(human_mask_binary > 0)
            if human_pixels > 0:
                # Calculate average confidence only from non-zero pixels in the original mask
                human_confidence = np.mean(human_mask[human_mask > 0]) if np.any(human_mask > 0) else 0.0
                human_confidence /= 255.0 # Normalize to 0.0-1.0

        # 各セグメントについて動的か判定
        for i, mask_data in enumerate(masks):
             # マスクデータが期待される形式か確認 (e.g., (H, W) or (1, H, W))
             if not isinstance(mask_data, np.ndarray):
                 logger.warning(f"セグメント {i} は NumPy 配列ではありません。スキップします。")
                 continue

             # 次元に応じて2Dマスクを取得
             if mask_data.ndim == 3 and mask_data.shape[0] == 1:
                 mask_2d = mask_data[0]
             elif mask_data.ndim == 2:
                 mask_2d = mask_data
             else:
                 logger.warning(f"セグメント {i} の次元 ({mask_data.shape}) が不正です。スキップします。")
                 continue

             # マスクをバイナリ化 (閾値 0.5)
             mask_binary = (mask_2d > 0.5).astype(np.uint8)
             segment_pixels = np.sum(mask_binary)
             if segment_pixels == 0: continue # 空のマスクはスキップ

             segment_ratio = segment_pixels / total_pixels

             # サイズフィルタリング
             if not (config.min_area_ratio <= segment_ratio <= config.max_area_ratio):
                 continue

             # --- 判定ロジック --- 
             is_dynamic = False
             segment_center = None # ループ開始時に初期化

             # 判定基準1: 人間マスクとの重なり
             if human_mask_binary is not None and human_pixels > 0:
                 human_overlap_mask = cv2.bitwise_and(human_mask_binary, mask_binary)
                 human_overlap_pixels = np.sum(human_overlap_mask)

                 if human_overlap_pixels > 0:
                     overlap_ratio_human = human_overlap_pixels / segment_pixels
                     # 人間マスクの信頼度に基づいて閾値を調整
                     adaptive_threshold_human = config.human_overlap_ratio * (1.0 - human_confidence * 0.3)
                     if overlap_ratio_human > adaptive_threshold_human:
                         is_dynamic = True

             # 判定基準2: 手の近くにあるか (動的と判定されていない場合のみ)
             if not is_dynamic and hand_positions:
                 segment_moments = cv2.moments(mask_binary)
                 if segment_moments["m00"] > 0:
                     cx = int(segment_moments["m10"] / segment_moments["m00"])
                     cy = int(segment_moments["m01"] / segment_moments["m00"])
                     segment_center = (cx, cy)

                 if segment_center:
                     min_distance_sq = float('inf')
                     for hand_x, hand_y in hand_positions:
                         distance_sq = (segment_center[0] - hand_x)**2 + (segment_center[1] - hand_y)**2
                         min_distance_sq = min(min_distance_sq, distance_sq)

                     min_distance = np.sqrt(min_distance_sq)

                     if min_distance < config.hand_proximity_threshold:
                          # 距離に応じて判定を緩和（設定された係数を使用）
                          # proximity_factor = config.hand_proximity_factor + (1.0 - config.hand_proximity_factor) * (min_distance / config.hand_proximity_threshold)
                          # => 手の近くのセグメントを動的と判定するロジックに変更
                          # 手の近くでも大きすぎるオブジェクトは除外 (e.g., 人間の体全体など)
                          if segment_ratio < config.max_area_ratio * 0.7: 
                               is_dynamic = True

             # 判定基準3: 動きマスクとの重なり (動的と判定されていない場合のみ)
             if not is_dynamic and total_motion_pixels > 0:
                 motion_overlap_mask = cv2.bitwise_and(motion_mask_binary, mask_binary)
                 motion_overlap_pixels = np.sum(motion_overlap_mask)

                 if motion_overlap_pixels > config.min_motion_pixels:
                     motion_overlap_ratio_seg = motion_overlap_pixels / segment_pixels

                     # セグメント中心部の動きを考慮
                     center_motion_ratio_seg = 0.0
                     if segment_center: # Use center calculated earlier if available
                         roi_size = int(np.sqrt(segment_pixels) * 0.3)
                         roi_size = max(10, min(50, roi_size))
                         cx, cy = segment_center
                         x1 = max(0, cx - roi_size)
                         y1 = max(0, cy - roi_size)
                         x2 = min(width, cx + roi_size) # Use width, not width-1
                         y2 = min(height, cy + roi_size) # Use height, not height-1

                         # Extract ROI using slicing for efficiency
                         center_mask_roi = mask_binary[y1:y2, x1:x2]
                         center_motion_roi = motion_mask_binary[y1:y2, x1:x2]
                         center_motion_in_mask = cv2.bitwise_and(center_mask_roi, center_motion_roi)

                         center_mask_pixels = np.sum(center_mask_roi)
                         if center_mask_pixels > 0:
                             center_motion_ratio_seg = np.sum(center_motion_in_mask) / center_mask_pixels

                     # 重なり率 or 中心部の重なり率が閾値を超えたら動的
                     if (motion_overlap_ratio_seg > config.motion_overlap_ratio or
                         center_motion_ratio_seg > config.center_motion_ratio):
                         is_dynamic = True

             # 最終判定
             if is_dynamic:
                 dynamic_indices.append(i)

    except Exception as e:
        logger.error(f"セグメントフィルタリングエラー: {e}", exc_info=True) # Include traceback

    return dynamic_indices
