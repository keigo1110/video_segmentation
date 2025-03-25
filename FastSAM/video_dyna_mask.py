"""
動的要素マスキングツール (DynaMask)

カメラの移動を考慮して、実世界で動いているオブジェクトのみをマスクするツール。
FastSAMセグメンテーション技術と組み合わせて使用します。
"""

import cv2
import numpy as np
import os
import logging
import sys
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

# 既存のセグメンテーションスクリプトをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import video_segment

# ライブラリの依存関係を処理
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8がインストールされていないため、人間検出の精度が低下する可能性があります。")
    print("pip install ultralytics でインストールしてください。")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipeがインストールされていないため、人間検出機能が無効になります。")
    print("pip install mediapipe でインストールしてください。")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DynaMask")


@dataclass
class Camera:
    """カメラの位置と姿勢情報"""
    image_id: str
    qw: float  # クォータニオン
    qx: float
    qy: float
    qz: float
    tx: float  # 平行移動
    ty: float
    tz: float
    camera_id: str
    name: str
    
    def get_rotation_matrix(self) -> np.ndarray:
        """クォータニオンから回転行列を計算"""
        r = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw])
        return r.as_matrix()
    
    def get_translation_vector(self) -> np.ndarray:
        """平行移動ベクトルを取得"""
        return np.array([self.tx, self.ty, self.tz])
    
    def get_projection_matrix(self, intrinsic: np.ndarray) -> np.ndarray:
        """射影行列を計算"""
        R = self.get_rotation_matrix()
        t = self.get_translation_vector()
        Rt = np.column_stack((R, t))
        return intrinsic @ Rt


@dataclass
class DynaMaskConfig:
    """DynaMaskの設定クラス"""
    # 入力設定
    input_type: str = "video"  # "video", "images", "camera"のいずれか
    input_path: Optional[str] = None  # 動画ファイルパスまたは画像フォルダパス
    poses_file: Optional[str] = None  # カメラ姿勢情報ファイル
    cameras_file: Optional[str] = None  # カメラ内部パラメータファイル
    image_pattern: str = "*.png"  # 画像シーケンスのパターン
    
    # セグメンテーション設定（FastSAM）
    fastsam_config: Optional[video_segment.Config] = None
    
    # 動き検出設定
    motion_threshold: float = 60.0  # 動きと判定するしきい値
    min_area_ratio: float = 0.01    # セグメント面積の最小比率（画像面積に対する割合）
    max_area_ratio: float = 0.4     # セグメント面積の最大比率
    temporal_consistency: int = 3   # 動的判定に必要な連続フレーム数
    blur_size: int = 5             # 前処理ブラーサイズ
    
    # 人間検出設定
    use_pose_detection: bool = True  # 人間のポーズ検出を使用するか
    pose_confidence: float = 0.4     # ポーズ検出の信頼度しきい値
    hand_confidence: float = 0.4     # 手の検出の信頼度しきい値
    hand_proximity_threshold: int = 70  # 手の近傍と判定する距離（ピクセル）
    use_yolo: bool = True           # YOLOv8を使用するか
    yolo_model: str = "yolov8n.pt"  # YOLOモデル名
    yolo_confidence: float = 0.3    # YOLO検出の信頼度しきい値
    
    # カメラキャリブレーション
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None  # 歪み係数
    use_homography: bool = True     # ホモグラフィー変換を使用するか
    camera_params: Dict[str, Dict] = field(default_factory=dict)  # カメラIDごとのパラメータ
    
    # 出力設定
    output_dir: Optional[str] = None  # 出力ディレクトリ
    output_filename: str = "dynamic_masked.mp4"  # 出力ファイル名
    save_debug_frames: bool = False  # デバッグ用フレームを保存するか
    debug_dir: Optional[str] = None  # デバッグフレーム保存ディレクトリ
    
    def __post_init__(self):
        """初期化後の処理"""
        # 出力ディレクトリの設定
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("output", f"dyna_{timestamp}")
        
        # デバッグディレクトリの設定
        if self.save_debug_frames and self.debug_dir is None:
            self.debug_dir = os.path.join(self.output_dir, "debug")
        
        # FastSAM設定の初期化
        if self.fastsam_config is None:
            self.fastsam_config = video_segment.Config()
            self.fastsam_config.input_type = self.input_type
            self.fastsam_config.input_path = self.input_path
            self.fastsam_config.image_pattern = self.image_pattern
            self.fastsam_config.output_dir = os.path.join(self.output_dir, "fastsam")


def read_camera_poses(poses_file: str) -> Dict[str, Camera]:
    """カメラの位置姿勢情報を読み込む

    Args:
        poses_file: 位置姿勢情報ファイルのパス（images.txt形式）

    Returns:
        画像IDをキーとするカメラ情報の辞書
    """
    cameras = {}
    
    with open(poses_file, 'r') as f:
        lines = f.readlines()
    
    # ヘッダー行をスキップ
    line_idx = 4  # 最初の実データ行
    
    while line_idx < len(lines):
        if not lines[line_idx].strip() or lines[line_idx].startswith('#'):
            line_idx += 1
            continue
            
        # カメラデータ行を読み込み
        camera_line = lines[line_idx].strip().split()
        
        # 十分なデータがあるか確認
        if len(camera_line) >= 10:
            try:
                image_id = camera_line[0]
                qw, qx, qy, qz = map(float, camera_line[1:5])
                tx, ty, tz = map(float, camera_line[5:8])
                camera_id = camera_line[8]
                name = ' '.join(camera_line[9:])
                
                cameras[image_id] = Camera(
                    image_id=image_id,
                    qw=qw, qx=qx, qy=qy, qz=qz,
                    tx=tx, ty=ty, tz=tz,
                    camera_id=camera_id,
                    name=name
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"カメラデータの解析エラー (行 {line_idx+1}): {e}")
        
        # 次のデータに進む（ポイントデータ行をスキップ）
        line_idx += 2
    
    logger.info(f"{len(cameras)}台のカメラ位置姿勢データを読み込みました")
    return cameras


def read_camera_parameters(cameras_file: str) -> Dict[str, Dict]:
    """カメラの内部パラメータを読み込む

    Args:
        cameras_file: カメラパラメータファイルのパス（COLMAP cameras.txt形式）

    Returns:
        カメラIDをキーとするカメラパラメータの辞書
    """
    camera_params = {}
    
    try:
        with open(cameras_file, 'r') as f:
            lines = f.readlines()
        
        # ヘッダー行をスキップ
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 4:  # 少なくともID、モデル、幅、高さが必要
                continue
            
            camera_id = parts[0]
            camera_model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            # カメラモデルに応じてパラメータを解析
            params = {}
            params["model"] = camera_model
            params["width"] = width
            params["height"] = height
            
            if camera_model in ["SIMPLE_PINHOLE", "PINHOLE"]:
                # SIMPLE_PINHOLE: f, cx, cy
                # PINHOLE: fx, fy, cx, cy
                if camera_model == "SIMPLE_PINHOLE" and len(parts) >= 7:
                    f = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    params["camera_matrix"] = np.array([
                        [f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    params["dist_coeffs"] = np.zeros(5, dtype=np.float32)
                
                elif camera_model == "PINHOLE" and len(parts) >= 8:
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    params["camera_matrix"] = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    params["dist_coeffs"] = np.zeros(5, dtype=np.float32)
            
            elif camera_model in ["SIMPLE_RADIAL", "RADIAL"]:
                # SIMPLE_RADIAL: f, cx, cy, k1
                # RADIAL: f, cx, cy, k1, k2
                if camera_model == "SIMPLE_RADIAL" and len(parts) >= 8:
                    f = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    k1 = float(parts[7])
                    params["camera_matrix"] = np.array([
                        [f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    params["dist_coeffs"] = np.array([k1, 0, 0, 0, 0], dtype=np.float32)
                
                elif camera_model == "RADIAL" and len(parts) >= 9:
                    f = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    k1 = float(parts[7])
                    k2 = float(parts[8])
                    params["camera_matrix"] = np.array([
                        [f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    params["dist_coeffs"] = np.array([k1, k2, 0, 0, 0], dtype=np.float32)
            
            camera_params[camera_id] = params
        
        logger.info(f"{len(camera_params)}台のカメラ内部パラメータを読み込みました")
    
    except Exception as e:
        logger.error(f"カメラパラメータの読み込みエラー: {e}")
    
    return camera_params


def compute_transformation_from_poses(prev_id: str, curr_id: str, 
                                   camera_poses: Dict[str, Camera], 
                                   config: DynaMaskConfig) -> np.ndarray:
    """カメラ位置情報から2フレーム間の変換行列を計算する"""
    if prev_id not in camera_poses or curr_id not in camera_poses:
        return np.eye(3, dtype=np.float32)
    
    prev_cam = camera_poses[prev_id]
    curr_cam = camera_poses[curr_id]
    
    # カメラの内部パラメータを取得
    camera_matrix = None
    
    # カメラパラメータが読み込まれている場合、そちらを優先して使用
    if prev_cam.camera_id in config.camera_params and "camera_matrix" in config.camera_params[prev_cam.camera_id]:
        camera_matrix = config.camera_params[prev_cam.camera_id]["camera_matrix"]
    elif config.camera_matrix is not None:
        camera_matrix = config.camera_matrix
    else:
        # 一般的なデフォルト値を使用
        h, w = config.fastsam_config.imgsz, config.fastsam_config.imgsz
        fx = fy = max(h, w)  # 焦点距離
        cx, cy = w/2, h/2    # 画像中心
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    # 各カメラの射影行列を計算
    P1 = prev_cam.get_projection_matrix(camera_matrix)
    P2 = curr_cam.get_projection_matrix(camera_matrix)
    
    # 平面に対する変換行列を計算
    normal = np.array([0, 0, 1])  # 地面を想定
    distance = 1.0
    
    # ホモグラフィー行列を計算
    R1 = prev_cam.get_rotation_matrix()
    R2 = curr_cam.get_rotation_matrix()
    t1 = prev_cam.get_translation_vector()
    t2 = curr_cam.get_translation_vector()
    
    # R2 * R1^T
    R_rel = R2 @ R1.T
    
    # t2 - R_rel * t1
    t_rel = t2 - R_rel @ t1
    
    # ホモグラフィー行列を構築
    H = camera_matrix @ (R_rel - (t_rel @ normal.T) / distance) @ np.linalg.inv(camera_matrix)
    
    return H.astype(np.float32)


def estimate_homography(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                       config: DynaMaskConfig, 
                       prev_id: Optional[str] = None,
                       curr_id: Optional[str] = None,
                       camera_poses: Optional[Dict[str, Camera]] = None) -> np.ndarray:
    """2フレーム間のホモグラフィー行列を推定する"""
    # カメラ位置情報が利用可能な場合はそれを使用
    if prev_id is not None and curr_id is not None and camera_poses is not None:
        H_poses = compute_transformation_from_poses(prev_id, curr_id, camera_poses, config)
        if not np.array_equal(H_poses, np.eye(3)):
            return H_poses
    
    # グレースケールに変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # ノイズ軽減のためにぼかす
    prev_gray = cv2.GaussianBlur(prev_gray, (config.blur_size, config.blur_size), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (config.blur_size, config.blur_size), 0)
    
    # 特徴点検出のためのパラメータ
    feature_params = dict(
        maxCorners=8000,
        qualityLevel=0.02,
        minDistance=12,
        blockSize=9,
    )
    
    # 特徴点を検出
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    
    # 特徴点が見つからない場合は単位行列を返す
    if prev_pts is None or len(prev_pts) < 8:
        logger.warning("十分な特徴点が見つかりませんでした。単位行列を使用します。")
        return np.eye(3, dtype=np.float32)
    
    # オプティカルフローで特徴点を追跡
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-4
    )
    
    # 追跡に成功した点のみを抽出
    if curr_pts is None:
        return np.eye(3, dtype=np.float32)
    
    # ステータスとエラーでフィルタリング
    mask = (status.ravel() == 1) & (err.ravel() < 10)
    if not np.any(mask):
        return np.eye(3, dtype=np.float32)
    
    prev_valid = prev_pts[mask]
    curr_valid = curr_pts[mask]
    
    # RANSACを使用してホモグラフィーを推定
    H, inliers = cv2.findHomography(
        prev_valid, curr_valid, 
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=5000,
        confidence=0.995
    )
    
    # ホモグラフィーの品質チェック
    if H is not None and inliers is not None:
        inlier_ratio = np.sum(inliers) / len(inliers)
        if inlier_ratio < 0.5:
            logger.warning(f"ホモグラフィーの信頼性が低いです（インライア比率: {inlier_ratio:.2f}）")
            return np.eye(3, dtype=np.float32)
        
        # ホモグラフィー行列の妥当性チェック
        if np.abs(np.linalg.det(H) - 1) > 0.2:
            logger.warning("不適切なホモグラフィー変換です")
            return np.eye(3, dtype=np.float32)
    else:
        return np.eye(3, dtype=np.float32)
    
    return H


def warp_frame(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """ホモグラフィー行列を使用してフレームをワープする"""
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, homography, (w, h))


def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                 homography: np.ndarray, config: DynaMaskConfig) -> np.ndarray:
    """カメラ移動を補正した後の動きを検出する"""
    # 前フレームをワープしてカメラ移動を補正
    warped_prev = warp_frame(prev_frame, homography)
    
    # グレースケールに変換して差分を計算
    prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # ノイズ軽減のためにぼかす
    blur_size = config.blur_size * 2 + 1
    prev_gray = cv2.GaussianBlur(prev_gray, (blur_size, blur_size), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (blur_size, blur_size), 0)
    
    # 複数のスケールで動きを検出
    mask = np.zeros_like(prev_gray)
    scales = [1.0, 0.75, 0.5]  # 複数のスケールで検出
    
    for scale in scales:
        if scale != 1.0:
            h, w = prev_gray.shape[:2]
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_prev = cv2.resize(prev_gray, (scaled_w, scaled_h))
            scaled_curr = cv2.resize(curr_gray, (scaled_w, scaled_h))
        else:
            scaled_prev = prev_gray
            scaled_curr = curr_gray
        
        # 絶対差分を計算
        diff = cv2.absdiff(scaled_prev, scaled_curr)
        
        # 適応的しきい値処理
        local_mask = cv2.adaptiveThreshold(
            diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            5
        )
        
        if scale != 1.0:
            local_mask = cv2.resize(local_mask, (prev_gray.shape[1], prev_gray.shape[0]))
        
        mask = cv2.bitwise_or(mask, local_mask)
    
    # モルフォロジー演算でノイズを除去
    kernel_open = np.ones((7, 7), np.uint8)
    kernel_close = np.ones((11, 11), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 連結成分の面積でフィルタリング
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)
    min_area = 100  # 最小面積（ピクセル数）
    
    for i in range(1, num_labels):  # 0はバックグラウンド
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    
    return filtered_mask


def detect_humans_and_hands(frame: np.ndarray, config: DynaMaskConfig, yolo_model=None) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """人間のポーズと手の位置を検出する"""
    height, width = frame.shape[:2]
    human_mask = np.zeros((height, width), dtype=np.uint8)
    hand_positions = []
    
    # MediaPipeが利用可能でない場合は空のマスクを返す
    if not MEDIAPIPE_AVAILABLE and not YOLO_AVAILABLE:
        return human_mask, hand_positions
    
    # RGB形式に変換（MediaPipeでは必要）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # YOLOv8による人間検出
    if config.use_yolo and YOLO_AVAILABLE and yolo_model is not None:
        try:
            # 人間の検出
            results = yolo_model(frame, verbose=False)[0]
            
            # 人間クラス (person = 0) の検出結果を処理
            for result in results.boxes.data:
                x1, y1, x2, y2, conf, cls = result
                
                # 人間クラスであり、閾値以上の信頼度を持つ場合
                if int(cls) == 0 and conf >= config.yolo_confidence:
                    # 座標を整数に変換
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # 人間領域をマスクに追加
                    cv2.rectangle(human_mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # 手の推定位置を追加
                    hand_height = y1 + (y2 - y1) // 2  # 人間の中間高さ
                    
                    # 左手と右手の推定位置
                    left_hand_x = x1
                    right_hand_x = x2
                    
                    hand_positions.append((left_hand_x, hand_height))
                    hand_positions.append((right_hand_x, hand_height))
                    
                    # 手の可能性がある領域を追加
                    hand_width = (x2 - x1) // 4
                    
                    # 左手エリア
                    left_hand_area = np.zeros_like(human_mask)
                    cv2.rectangle(left_hand_area, 
                                 (max(0, x1 - hand_width), max(0, y1)), 
                                 (x1 + hand_width, y2), 
                                 255, -1)
                    
                    # 右手エリア
                    right_hand_area = np.zeros_like(human_mask)
                    cv2.rectangle(right_hand_area, 
                                 (x2 - hand_width, max(0, y1)), 
                                 (min(width, x2 + hand_width), y2), 
                                 255, -1)
                    
                    # マスクに手の領域を追加
                    human_mask = cv2.bitwise_or(human_mask, left_hand_area)
                    human_mask = cv2.bitwise_or(human_mask, right_hand_area)
        
        except Exception as e:
            logger.error(f"YOLOによる人間検出エラー: {e}")
    
    # MediaPipeが使用可能で、ポーズ検出が有効な場合
    if MEDIAPIPE_AVAILABLE and config.use_pose_detection:
        # MediaPipeによるポーズ検出
        pose_configs = [
            {"model_complexity": 1, "min_detection_confidence": config.pose_confidence},
            {"model_complexity": 2, "min_detection_confidence": config.pose_confidence - 0.1}
        ]
        
        pose_detected = False
        for pose_config in pose_configs:
            if pose_detected:
                break
                
            with mp_pose.Pose(
                static_image_mode=False,
                **pose_config
            ) as pose:
                pose_results = pose.process(rgb_frame)
                
                if pose_results.pose_landmarks:
                    pose_detected = True
                    # ランドマークを抽出
                    landmarks = pose_results.pose_landmarks.landmark
                    
                    # 人間の位置を多角形として描画
                    points = []
                    key_indices = [0, 11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28]
                    
                    for idx in key_indices:
                        if landmarks[idx].visibility > 0.3:
                            x, y = int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                            points.append([x, y])
                    
                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        
                        if len(points) >= 3:
                            cv2.fillPoly(human_mask, [points], 255)
                        else:
                            for point in points:
                                cv2.circle(human_mask, tuple(point), 30, 255, -1)
                        
                        # 全身マスクを作成
                        all_visible_points = []
                        for i, landmark in enumerate(landmarks):
                            if landmark.visibility > 0.2:
                                x, y = int(landmark.x * width), int(landmark.y * height)
                                all_visible_points.append([x, y])
                        
                        if len(all_visible_points) > 2:
                            all_visible_points = np.array(all_visible_points, dtype=np.int32)
                            
                            if len(all_visible_points) >= 3:
                                hull = cv2.convexHull(all_visible_points)
                                cv2.fillPoly(human_mask, [hull], 255)
                            else:
                                for i in range(len(all_visible_points) - 1):
                                    pt1 = tuple(all_visible_points[i])
                                    pt2 = tuple(all_visible_points[i + 1])
                                    cv2.line(human_mask, pt1, pt2, 255, 5)
                    
                    # 手の位置を特定
                    for hand_idx in [15, 16, 17, 18, 19, 20]:
                        if landmarks[hand_idx].visibility > 0.3:
                            x, y = int(landmarks[hand_idx].x * width), int(landmarks[hand_idx].y * height)
                            hand_positions.append((x, y))
                            cv2.circle(human_mask, (x, y), config.hand_proximity_threshold, 255, -1)
        
        # 手の詳細検出
        hands_configs = [
            {"max_num_hands": 4, "min_detection_confidence": config.hand_confidence},
            {"max_num_hands": 4, "min_detection_confidence": config.hand_confidence - 0.1}
        ]
        
        for hands_config in hands_configs:
            with mp_hands.Hands(
                static_image_mode=False,
                **hands_config
            ) as hands:
                hands_results = hands.process(rgb_frame)
                
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        hand_points = []
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            hand_points.append([x, y])
                        
                        if hand_points:
                            hand_points = np.array(hand_points, dtype=np.int32)
                            M = cv2.moments(hand_points)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                hand_positions.append((cx, cy))
                            
                            if len(hand_points) >= 3:
                                hull = cv2.convexHull(hand_points)
                                cv2.fillPoly(human_mask, [hull], 255)
                                
                                hand_area_dilated = np.zeros_like(human_mask)
                                cv2.fillPoly(hand_area_dilated, [hull], 255)
                                kernel = np.ones((15, 15), np.uint8)
                                hand_area_dilated = cv2.dilate(hand_area_dilated, kernel)
                                human_mask = cv2.bitwise_or(human_mask, hand_area_dilated)
                            else:
                                for point in hand_points:
                                    cv2.circle(human_mask, tuple(point), 20, 255, -1)
    
    # マスクの拡張
    kernel = np.ones((9, 9), np.uint8)
    human_mask = cv2.dilate(human_mask, kernel, iterations=3)
    
    return human_mask, hand_positions


def filter_dynamic_segments(segments: Any, motion_mask: np.ndarray, 
                           frame_shape: Tuple[int, int], config: DynaMaskConfig,
                           human_mask: Optional[np.ndarray] = None,
                           hand_positions: Optional[List[Tuple[int, int]]] = None) -> List[int]:
    """動的なセグメントのインデックスをフィルタリングする"""
    dynamic_indices = []
    height, width = frame_shape[:2]
    total_pixels = height * width
    
    try:
        # セグメントマスクの取得
        if hasattr(segments, "__getitem__") and len(segments) > 0:
            if hasattr(segments[0], "masks") and segments[0].masks is not None:
                masks = segments[0].masks.data.cpu().numpy()
            else:
                return []
        elif hasattr(segments, "masks") and segments.masks is not None:
            masks = segments.masks.data.cpu().numpy() if hasattr(segments.masks, "data") else segments.masks
        else:
            return []
        
        # 各セグメントについて動的か判定
        for i, mask in enumerate(masks):
            # マスクの次元に応じた処理
            if isinstance(mask, np.ndarray) and mask.ndim > 1:
                mask_2d = mask[0] if mask.ndim > 2 else mask
            else:
                continue
            
            # セグメントのピクセル数を計算
            segment_pixels = np.sum(mask_2d > 0.5)
            segment_ratio = segment_pixels / total_pixels
            
            # サイズでフィルタリング
            if segment_ratio < config.min_area_ratio or segment_ratio > config.max_area_ratio:
                continue
            
            # セグメントと動きマスクの重なりを計算
            mask_binary = (mask_2d > 0.5).astype(np.uint8) * 255
            overlap = cv2.bitwise_and(motion_mask, mask_binary)
            overlap_ratio = np.sum(overlap > 0) / max(1, segment_pixels)
            
            # セグメントの中心と周辺部の動きを分析
            moments = cv2.moments(mask_binary)
            is_dynamic = False
            
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # 中心領域の動きを確認
                center_region_size = 20
                center_region = motion_mask[
                    max(0, cy-center_region_size):min(height, cy+center_region_size),
                    max(0, cx-center_region_size):min(width, cx+center_region_size)
                ]
                center_motion = np.mean(center_region) if center_region.size > 0 else 0
                
                # エッジ部分の動きを確認
                edge_kernel = np.ones((3, 3), np.uint8)
                edge_mask = cv2.dilate(mask_binary, edge_kernel) - cv2.erode(mask_binary, edge_kernel)
                edge_motion = cv2.bitwise_and(motion_mask, edge_mask)
                edge_ratio = np.sum(edge_motion > 0) / max(1, np.sum(edge_mask > 0))
                
                # 動的判定の条件
                standard_dynamic_condition = (
                    overlap_ratio > 0.5 and
                    center_motion > 60 and
                    edge_ratio < 0.7
                )
                
                # 人間マスクとの重なりを確認
                human_overlap = False
                near_hand = False
                
                if human_mask is not None:
                    # 人間との重なりを計算
                    human_overlap_mask = cv2.bitwise_and(human_mask, mask_binary)
                    human_overlap_ratio = np.sum(human_overlap_mask > 0) / max(1, segment_pixels)
                    
                    if human_overlap_ratio > 0.2:
                        human_overlap = True
                
                # 手の近くにあるか確認
                if hand_positions:
                    for hand_x, hand_y in hand_positions:
                        # セグメントの中心と手の距離を計算
                        distance = np.sqrt((cx - hand_x)**2 + (cy - hand_y)**2)
                        
                        if distance < config.hand_proximity_threshold:
                            near_hand = True
                            break
                
                # 動的判定
                if standard_dynamic_condition:
                    is_dynamic = True
                elif human_overlap:
                    is_dynamic = True
                elif near_hand:
                    relaxed_condition = (
                        overlap_ratio > 0.2 and
                        center_motion > 20
                    )
                    if relaxed_condition:
                        is_dynamic = True
            
            if is_dynamic:
                dynamic_indices.append(i)
    
    except Exception as e:
        logger.error(f"セグメントフィルタリングエラー: {e}")
    
    return dynamic_indices


def process_video_with_dynamic_masking(config: DynaMaskConfig) -> str:
    """動的要素のみをマスクする処理を実行する"""
    # 出力ディレクトリの作成
    os.makedirs(config.output_dir, exist_ok=True)
    if config.save_debug_frames:
        os.makedirs(config.debug_dir, exist_ok=True)
    
    # FastSAMモデルをロード
    fastsam_model = video_segment.load_model(config.fastsam_config)
    
    # YOLOv8モデルをロード（人間検出用）
    yolo_model = None
    if config.use_yolo and YOLO_AVAILABLE:
        try:
            yolo_model = YOLO(config.yolo_model)
            logger.info(f"YOLOv8モデルをロードしました: {config.yolo_model}")
        except Exception as e:
            logger.error(f"YOLOv8モデルのロードエラー: {e}")
            config.use_yolo = False
    
    # 入力ソースの初期化
    source = video_segment.initialize_input_source(config.fastsam_config)
    width, height = source["width"], source["height"]
    fps = source["fps"]
    
    # 出力動画の設定
    output_path = os.path.join(config.output_dir, config.output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # カメラ姿勢情報を読み込み
    camera_poses = {}
    if config.poses_file and os.path.exists(config.poses_file):
        try:
            camera_poses = read_camera_poses(config.poses_file)
        except Exception as e:
            logger.error(f"カメラ姿勢情報の読み込みエラー: {e}")
    
    # カメラ内部パラメータを読み込み
    if config.cameras_file and os.path.exists(config.cameras_file):
        try:
            config.camera_params = read_camera_parameters(config.cameras_file)
            
            # デフォルトのカメラ行列を設定
            if config.camera_params and config.camera_matrix is None:
                first_camera = next(iter(config.camera_params.values()))
                if "camera_matrix" in first_camera:
                    config.camera_matrix = first_camera["camera_matrix"]
                    logger.info(f"デフォルトカメラ行列を設定: {config.camera_matrix}")
                
                if "dist_coeffs" in first_camera:
                    config.dist_coeffs = first_camera["dist_coeffs"]
        except Exception as e:
            logger.error(f"カメラ内部パラメータの読み込みエラー: {e}")
    
    # 処理メイン部分
    prev_frame = None
    prev_result = None
    dynamic_history = {}  # セグメントの動的履歴を追跡
    
    frame_idx = 0
    logger.info("動的マスキング処理を開始します...")
    
    try:
        while True:
            # フレームを取得
            ret, frame = video_segment.get_next_frame(source, frame_idx)
            if not ret:
                break
            
            # 最初のフレームの場合は初期化のみ
            if prev_frame is None:
                prev_frame = frame.copy()
                # 最初のフレームのセグメント化
                prev_result = video_segment.predict_with_model(
                    fastsam_model, prev_frame, config.fastsam_config
                )
                frame_idx += 1
                continue
            
            # カメラ移動補正のためのホモグラフィー推定
            H = estimate_homography(prev_frame, frame, config)
            
            # 動きマスクの検出
            motion_mask = detect_motion(prev_frame, frame, H, config)
            
            # 人間と手の検出
            human_mask = None
            hand_positions = []
            if config.use_pose_detection:
                human_mask, hand_positions = detect_humans_and_hands(frame, config, yolo_model)
                
                # デバッグ用：人間検出結果を可視化
                if config.save_debug_frames:
                    human_vis = cv2.cvtColor(human_mask, cv2.COLOR_GRAY2BGR)
                    human_debug_path = os.path.join(config.debug_dir, f"human_{frame_idx:06d}.png")
                    cv2.imwrite(human_debug_path, human_vis)
                    
                    # 手の位置をマーク
                    hand_vis = frame.copy()
                    for hx, hy in hand_positions:
                        cv2.circle(hand_vis, (hx, hy), config.hand_proximity_threshold, (0, 255, 0), 2)
                    hand_debug_path = os.path.join(config.debug_dir, f"hands_{frame_idx:06d}.png")
                    cv2.imwrite(hand_debug_path, hand_vis)
            
            # 現在フレームをセグメント化
            curr_result = video_segment.predict_with_model(
                fastsam_model, frame, config.fastsam_config
            )
            
            # 動的セグメントをフィルタリング
            dynamic_indices = filter_dynamic_segments(
                curr_result, motion_mask, frame.shape, config,
                human_mask, hand_positions
            )
            
            # 時間的一貫性を考慮して動的判定を更新
            for idx in dynamic_indices:
                if idx not in dynamic_history:
                    dynamic_history[idx] = 0
                dynamic_history[idx] += 1
            
            # 各セグメントの動的状態を判定
            final_dynamic_indices = [
                idx for idx, count in dynamic_history.items() 
                if count >= config.temporal_consistency
            ]
            
            # 動的セグメントのみをマスクした結果を作成
            output_frame = frame.copy()
            
            try:
                # マスクの取得
                masks = None
                if hasattr(curr_result, "__getitem__") and len(curr_result) > 0:
                    if hasattr(curr_result[0], "masks") and curr_result[0].masks is not None:
                        masks = curr_result[0].masks.data.cpu().numpy()
                elif hasattr(curr_result, "masks") and curr_result.masks is not None:
                    masks = curr_result.masks.data.cpu().numpy() if hasattr(curr_result.masks, "data") else curr_result.masks
                
                if masks is not None and len(masks) > 0:
                    # 動的セグメントのカラー設定
                    colors = np.random.randint(50, 255, size=(len(final_dynamic_indices), 3), dtype=np.uint8)
                    
                    # デバッグ用：動きマスクを可視化
                    if config.save_debug_frames:
                        motion_vis = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                        debug_path = os.path.join(config.debug_dir, f"motion_{frame_idx:06d}.png")
                        cv2.imwrite(debug_path, motion_vis)
                    
                    # 各動的セグメントを描画
                    for i, idx in enumerate(final_dynamic_indices):
                        if idx < len(masks):
                            mask = masks[idx]
                            mask_2d = mask[0] if mask.ndim > 2 else mask
                            
                            # マスクのオーバーレイ
                            color = colors[i % len(colors)].tolist()
                            mask_image = np.zeros_like(frame, dtype=np.uint8)
                            mask_image[mask_2d > 0.5] = color
                            output_frame = cv2.addWeighted(
                                output_frame, 1.0, mask_image, 0.5, 0
                            )
                            
                            # マスクの輪郭を描画
                            contour_mask = (mask_2d > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(
                                contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(output_frame, contours, -1, color, 2)
                    
                    # 動的セグメント数を表示
                    cv2.putText(
                        output_frame,
                        f"Dynamic Objects: {len(final_dynamic_indices)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
            
            except Exception as e:
                logger.error(f"マスク処理エラー: {e}")
            
            # 結果を保存
            out.write(output_frame)
            
            # 1秒ごとにフレームも保存
            if frame_idx % int(fps) == 0:
                frame_path = os.path.join(config.output_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, output_frame)
            
            # 結果を表示（必要に応じて）
            if config.fastsam_config.show_frames:
                display_scale = config.fastsam_config.display_scale
                display_width = int(width * display_scale)
                display_height = int(height * display_scale)
                cv2.imshow(
                    "DynaMask",
                    cv2.resize(output_frame, (display_width, display_height)),
                )
                
                # キー入力の処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            
            # 次のフレームに更新
            prev_frame = frame.copy()
            prev_result = curr_result
            
            # 一定間隔で動的履歴の古いエントリをクリア
            if frame_idx % 10 == 0:
                dynamic_history = {
                    idx: count for idx, count in dynamic_history.items()
                    if count > 0
                }
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                logger.info(f"フレーム {frame_idx} 処理完了...")
    
    except KeyboardInterrupt:
        logger.info("処理が中断されました")
    finally:
        # リソースを解放
        if source["cap"] is not None:
            source["cap"].release()
        out.release()
        cv2.destroyAllWindows()
    
    logger.info(f"処理完了: {frame_idx} フレーム処理されました")
    logger.info(f"出力ファイル: {output_path}")
    
    return config.output_dir


def parse_arguments():
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(
        description="カメラ移動を考慮して動的要素のみをマスクするツール",
    )
    
    # 入力ソースの指定
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="入力動画ファイルのパス")
    input_group.add_argument("--images", type=str, help="入力画像シーケンスのフォルダパス")
    input_group.add_argument("--camera", type=int, default=None, help="カメラデバイスID（通常は0）")
    
    # 画像シーケンス設定
    parser.add_argument("--image-pattern", type=str, default="*.png", help="画像シーケンスのパターン（例: *.png, *.jpg）")
    
    # カメラ情報
    parser.add_argument("--poses", type=str, default=None, help="カメラ位置姿勢情報ファイル（images.txt形式）")
    parser.add_argument("--cameras", type=str, default=None, help="カメラ内部パラメータファイル（cameras.txt形式）")
    
    # 動き検出設定
    parser.add_argument("--motion-threshold", type=float, default=30.0, help="動きと判定するしきい値")
    parser.add_argument("--min-area", type=float, default=0.01, help="セグメント面積の最小比率")
    parser.add_argument("--max-area", type=float, default=0.7, help="セグメント面積の最大比率")
    parser.add_argument("--temporal", type=int, default=3, help="動的判定に必要な連続フレーム数")
    
    # 人間検出設定
    parser.add_argument("--no-yolo", dest="use_yolo", action="store_false", help="YOLOによる人間検出を無効化")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLOモデル名")
    parser.add_argument("--no-pose", dest="use_pose", action="store_false", help="MediaPipeによるポーズ検出を無効化")
    
    # FastSAM設定
    parser.add_argument("--model", type=str, default="FastSAM-x", help="モデル名")
    parser.add_argument("--conf", type=float, default=0.7, help="信頼度しきい値")
    
    # 出力設定
    parser.add_argument("--output", type=str, default=None, help="出力ディレクトリ")
    parser.add_argument("--save-debug", action="store_true", help="デバッグフレームを保存する")
    parser.add_argument("--no-show", dest="show", action="store_false", help="フレームを表示しない")
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    # 入力タイプの設定
    input_type = None
    input_path = None
    camera_id = 0
    
    if args.video:
        input_type = "video"
        input_path = args.video
    elif args.images:
        input_type = "images"
        input_path = args.images
    elif args.camera is not None:
        input_type = "camera"
        camera_id = args.camera
    
    # FastSAM設定
    fastsam_config = video_segment.Config(
        input_type=input_type,
        input_path=input_path,
        camera_id=camera_id,
        model_name=args.model,
        confidence=args.conf,
        show_frames=args.show,
        image_pattern=args.image_pattern,
        auto_size=True,
        imgsz=1024
    )
    
    # DynaMask設定
    config = DynaMaskConfig(
        input_type=input_type,
        input_path=input_path,
        poses_file=args.poses,
        cameras_file=args.cameras,
        fastsam_config=fastsam_config,
        motion_threshold=60.0,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        temporal_consistency=args.temporal,
        blur_size=9,
        image_pattern=args.image_pattern,
        use_pose_detection=args.use_pose,
        pose_confidence=0.4,
        hand_confidence=0.4,
        hand_proximity_threshold=70,
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        output_dir=args.output,
        save_debug_frames=args.save_debug
    )
    
    # 処理実行
    process_video_with_dynamic_masking(config)


if __name__ == "__main__":
    print("=" * 60)
    print(" DynaMask - 動的要素マスキングツール")
    print("=" * 60)
    print("カメラの移動を考慮して実世界で動いているオブジェクトのみをマスクします")
    print("使用方法: python dynamask.py --video input.mp4 [--poses camera_poses.txt] [--cameras cameras.txt]")
    print("=" * 60)
    
    main()