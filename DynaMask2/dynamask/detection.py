import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DynaMaskConfig

logger = logging.getLogger(__name__)

# ライブラリの依存関係を条件付きで処理
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    # logger.info は複数回表示される可能性があるので、ここでは警告レベルにするか、アプリケーション起動時に一度だけ表示するのが望ましい
    # logger.warning("YOLOv8 がインストールされていません。人間検出の精度が低下する可能性があります。")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    # logger.warning("MediaPipe がインストールされていません。人間検出機能が無効になります。")

def detect_humans_and_hands(frame: np.ndarray, config: 'DynaMaskConfig', yolo_model: Optional[Any] = None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """人間のポーズと手の位置を検出する - 改良版アルゴリズム

    Args:
        frame: 入力フレーム
        config: 設定
        yolo_model: YOLOモデル (Optional)

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]]]: 人間マスクと手の位置リスト
    """
    height, width = frame.shape[:2]
    human_mask = np.zeros((height, width), dtype=np.uint8)
    hand_positions = []

    # 依存ライブラリが利用できない場合は警告を出し、空の結果を返す
    if not config.use_pose_detection and not config.use_yolo:
         logger.warning("人間検出が無効化されています (use_pose_detection=False, use_yolo=False)。")
         return human_mask, hand_positions
    if config.use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO が設定で有効になっていますが、ライブラリが見つかりません。YOLO検出をスキップします。")
    if config.use_pose_detection and not MEDIAPIPE_AVAILABLE:
        logger.warning("MediaPipe が設定で有効になっていますが、ライブラリが見つかりません。MediaPipe検出をスキップします。")


    # 確実な人間検出のためのマルチステージアプローチ実装

    # ステージ1: YOLOによる高信頼度の人間検出
    reliable_human_boxes = []
    yolo_mask = np.zeros((height, width), dtype=np.uint8)

    if config.use_yolo and YOLO_AVAILABLE and yolo_model is not None:
        try:
            # Use predict method for standard Ultralytics interface
            results = yolo_model.predict(frame, verbose=False, conf=config.yolo_confidence)[0]

            has_masks = hasattr(results, 'masks') and results.masks is not None

            if has_masks:
                # セグメントベース
                for i, mask_tensor in enumerate(results.masks.data):
                    if i >= len(results.boxes.data): continue # Ensure index is valid
                    box = results.boxes.data[i]
                    # Ensure box has expected number of elements
                    if len(box) < 6: continue
                    cls, conf = int(box[5]), float(box[4])

                    if cls == 0 and conf >= config.yolo_confidence:
                        x1, y1, x2, y2 = map(int, box[:4])
                        reliable_human_boxes.append((x1, y1, x2, y2, conf))

                        mask_np = mask_tensor.cpu().numpy()
                        mask_binary = (mask_np > 0.5).astype(np.uint8)

                        confidence_factor = min(1.0, conf / 0.8)
                        mask_value = int(200 * confidence_factor) + 55

                        # Use maximum to handle overlaps correctly
                        yolo_mask[mask_binary > 0] = np.maximum(yolo_mask[mask_binary > 0], mask_value)
            elif hasattr(results, 'boxes') and results.boxes is not None:
                 # バウンディングボックスベース
                 for box in results.boxes.data:
                     if len(box) < 6: continue
                     x1, y1, x2, y2, conf, cls = box
                     if int(cls) == 0 and conf >= config.yolo_confidence:
                         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                         reliable_human_boxes.append((x1, y1, x2, y2, float(conf)))

                         confidence_factor = min(1.0, float(conf) / 0.8)
                         mask_value = int(200 * confidence_factor) + 55
                         # Fill rectangle, potentially overwriting; consider max logic if needed
                         cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), mask_value, -1)

        except Exception as e:
            logger.error(f"YOLOによる人間検出エラー: {e}", exc_info=True) # Add traceback

    # ステージ2: MediaPipeによるポーズ検出
    mediapipe_mask = np.zeros((height, width), dtype=np.uint8)
    pose_landmarks_detected = False
    rgb_frame = None
    if config.use_pose_detection and MEDIAPIPE_AVAILABLE:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
             logger.error(f"フレームのRGB変換エラー: {e}. MediaPipe検出をスキップします。")

    if config.use_pose_detection and MEDIAPIPE_AVAILABLE and rgb_frame is not None:
        try:
            # ポーズ検出
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1, # Consider making this configurable
                min_detection_confidence=config.pose_confidence,
                min_tracking_confidence=0.5
            ) as pose:
                pose_results = pose.process(rgb_frame)

                if pose_results.pose_landmarks:
                    pose_landmarks_detected = True
                    landmarks = pose_results.pose_landmarks.landmark
                    valid_points, torso_points, face_points, arm_points = [], [], [], []

                    for i, landmark in enumerate(landmarks):
                        # Check visibility and presence robustly
                        visible = landmark.HasField('visibility') and landmark.visibility >= 0.3
                        present = landmark.HasField('x') and landmark.HasField('y')
                        if not (visible or present): continue # Skip if neither visible nor present

                        x, y = int(landmark.x * width), int(landmark.y * height)
                        x = max(0, min(width - 1, x)) # Clamp coordinates
                        y = max(0, min(height - 1, y))

                        # グループ分け (可視性閾値も考慮)
                        vis_threshold = landmark.visibility if landmark.HasField('visibility') else 0.3 # Default if no visibility
                        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: # 顔
                            if vis_threshold > 0.6: face_points.append((x, y)); valid_points.append((x, y))
                        elif i in [11, 12, 23, 24]: # 胴体
                            if vis_threshold > 0.4: torso_points.append((x, y)); valid_points.append((x, y))
                        elif i in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]: # 腕/手
                            if vis_threshold > 0.3: arm_points.append((x, y)); valid_points.append((x, y))
                        else: # 脚なども含める場合
                             valid_points.append((x, y)) # Add other points to valid_points

                    if len(valid_points) >= 5:
                        # 手の位置候補を追加 (顔と腕の末端)
                        hand_positions.extend(face_points)
                        for idx in [16, 18, 20, 22]: # 手首/指先
                            if idx < len(landmarks):
                                lm = landmarks[idx]
                                vis = lm.visibility if lm.HasField('visibility') else 0.0
                                if lm.HasField('x') and lm.HasField('y') and vis > 0.5:
                                     x = max(0, min(width - 1, int(lm.x * width)))
                                     y = max(0, min(height - 1, int(lm.y * height)))
                                     hand_positions.append((x, y))

                        # マスク生成 (Convex Hull)
                        try:
                            if len(torso_points) >= 3: # Convex hull needs >= 3 points
                                hull = cv2.convexHull(np.array(torso_points, dtype=np.int32))
                                cv2.fillPoly(mediapipe_mask, [hull], 150)
                            if len(valid_points) >= 3:
                                hull = cv2.convexHull(np.array(valid_points, dtype=np.int32))
                                # Use maximum to avoid overwriting torso mask
                                temp_mask = np.zeros_like(mediapipe_mask)
                                cv2.fillPoly(temp_mask, [hull], 100)
                                mediapipe_mask = np.maximum(mediapipe_mask, temp_mask)
                        except Exception as hull_error:
                            logger.warning(f"MediaPipe Pose Convex Hull計算エラー: {hull_error}")

            # 手の検出
            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=config.hand_confidence,
                min_tracking_confidence=0.5
            ) as hands:
                hands_results = hands.process(rgb_frame)

                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        hand_points, hand_center_x, hand_center_y, valid_landmarks = [], 0.0, 0.0, 0
                        for landmark in hand_landmarks.landmark:
                            if not landmark.HasField('x') or not landmark.HasField('y'): continue
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            if 0 <= x < width and 0 <= y < height:
                                hand_points.append([x, y])
                                hand_center_x += x
                                hand_center_y += y
                                valid_landmarks += 1

                        if valid_landmarks >= 10: # 十分なランドマーク
                            if len(hand_points) >= 3: # Convex hull needs >= 3 points
                                try:
                                    hull = cv2.convexHull(np.array(hand_points, dtype=np.int32))
                                    # Use maximum for overlap
                                    temp_mask = np.zeros_like(mediapipe_mask)
                                    cv2.fillPoly(temp_mask, [hull], 200) # 手は強めのマスク
                                    mediapipe_mask = np.maximum(mediapipe_mask, temp_mask)
                                except Exception as hull_error:
                                    logger.warning(f"MediaPipe Hand Convex Hull計算エラー: {hull_error}")

                            if valid_landmarks > 0:
                                hand_center = (int(hand_center_x / valid_landmarks), int(hand_center_y / valid_landmarks))
                                hand_positions.append(hand_center)
        except Exception as e:
            logger.error(f"MediaPipeによる検出エラー: {e}", exc_info=True) # Add traceback


    # ステージ3: コンテキスト分析と検証
    yolo_binary_mask = (yolo_mask > 100).astype(np.uint8)
    mediapipe_binary_mask = (mediapipe_mask > 0).astype(np.uint8)
    both_detected_binary = cv2.bitwise_and(yolo_binary_mask, mediapipe_binary_mask)

    if pose_landmarks_detected:
        if np.any(both_detected_binary):
            # 両方で検出された場合、両方のマスクの最大値を取る
            human_mask = np.maximum(yolo_mask, mediapipe_mask)
        else:
            # 一致しない場合、YOLOの高信頼度 (>150) のみ採用
            human_mask = (yolo_mask > 150).astype(np.uint8) * 255
    else:
        # ポーズがない場合、YOLOの超高信頼度 (>200) のみ採用
        human_mask = (yolo_mask > 200).astype(np.uint8) * 255

    # 膨張処理 (ステージ4の後の方が良いかもしれない)
    # kernel_dilate = np.ones((5, 5), np.uint8)
    # human_mask = cv2.dilate(human_mask, kernel_dilate, iterations=1)

    # ステージ4: 手の位置の精密検出と検証済み手のマスク追加
    validated_hand_positions = []
    temp_hand_mask = np.zeros_like(human_mask) # Mask for validated hands

    if hand_positions:
        clustered_hands = []
        # クラスタリング (手の位置候補をまとめる)
        for hx, hy in hand_positions:
            if not (0 <= hx < width and 0 <= hy < height): continue
            assigned = False
            for i, cluster in enumerate(clustered_hands):
                 if len(cluster) == 4:
                    points, cx, cy, count = cluster
                    try:
                        dist = np.sqrt((hx - cx)**2 + (hy - cy)**2)
                        if dist < config.hand_proximity_threshold * 0.5: # Use multiplier of threshold
                            points.append((hx, hy))
                            new_cx = (cx * count + hx) / (count + 1)
                            new_cy = (cy * count + hy) / (count + 1)
                            clustered_hands[i] = (points, new_cx, new_cy, count + 1)
                            assigned = True
                            break
                    except TypeError as e:
                        logger.error(f"手のクラスタリング距離計算エラー: {e}. Cluster: {cluster}, Point: {(hx, hy)}")
                        continue # Skip this point for this cluster
                 else:
                      logger.warning(f"不正な形式のクラスタ: {cluster}")
                      continue

            if not assigned:
                clustered_hands.append(([(hx, hy)], float(hx), float(hy), 1))

        # クラスタのスコアリングと検証
        human_mask_binary_for_overlap = (human_mask > 0).astype(np.uint8) # Pre-calculate binary mask
        for cluster in clustered_hands:
            if len(cluster) == 4:
                points, cx, cy, count = cluster
                center_x_int, center_y_int = int(cx), int(cy)
                if not (0 <= center_x_int < width and 0 <= center_y_int < height): continue # Skip if center is out of bounds

                point_score = min(1.0, count / 5.0)
                # 人間マスクとの重なりスコア
                hand_area = np.zeros((height, width), dtype=np.uint8)
                radius = config.hand_proximity_threshold // 2
                cv2.circle(hand_area, (center_x_int, center_y_int), radius, 255, -1)
                overlap = cv2.bitwise_and(hand_area, human_mask_binary_for_overlap)
                overlap_pixels = np.sum(overlap > 0)
                overlap_score = min(1.0, overlap_pixels / (np.pi * radius**2 + 1e-6)) # Normalize by circle area

                total_score = point_score * 0.6 + overlap_score * 0.4
                if total_score > 0.3 or overlap_score > 0.8:
                    validated_hand_positions.append((center_x_int, center_y_int))
                    # 検証済みの手の位置にマスクを追加
                    cv2.circle(temp_hand_mask, (center_x_int, center_y_int), config.hand_proximity_threshold, 255, -1)
            else:
                 logger.warning(f"スコアリング中の不正な形式のクラスタ: {cluster}")


    validated_hand_positions = validated_hand_positions[:3] # 最大3つに制限

    # 人間マスクと検証済み手のマスクを結合
    human_mask = cv2.bitwise_or(human_mask, temp_hand_mask)

    # ステージ5: 最終的なマスク生成（ノイズ除去）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # Apply opening first to remove small noise, then closing to fill gaps
    human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 小さすぎる連結成分を除去
    contours, _ = cv2.findContours(human_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = height * width * 0.001 # Reduce threshold slightly? Or make configurable?
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                cv2.drawContours(human_mask, [contour], -1, 0, -1)

    # フォールバック (YOLO検出結果がある場合)
    if np.sum(human_mask) == 0 and reliable_human_boxes:
        try:
            best_box = max(reliable_human_boxes, key=lambda box: box[4])
            x1, y1, x2, y2, _ = map(int, best_box[:5])
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            # Draw rectangle within bounds
            pt1 = (max(0, x1 + margin_x), max(0, y1 + margin_y))
            pt2 = (min(width - 1, x2 - margin_x), min(height - 1, y2 - margin_y))
            # Ensure pt1 < pt2
            if pt1[0] < pt2[0] and pt1[1] < pt2[1]:
                 cv2.rectangle(human_mask, pt1, pt2, 255, -1)
        except (ValueError, IndexError):
            logger.error("フォールバック用の信頼できる人間ボックスの処理中にエラー発生。", exc_info=True)
        except Exception as e:
            logger.error(f"フォールバック処理中に予期せぬエラー: {e}", exc_info=True)

    return human_mask, validated_hand_positions
