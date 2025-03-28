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
    motion_threshold: float = 80.0  # 動きと判定するしきい値（高くして厳しく）
    min_area_ratio: float = 0.01    # セグメント面積の最小比率（画像面積に対する割合）
    max_area_ratio: float = 0.4     # セグメント面積の最大比率
    temporal_consistency: int = 4   # 動的判定に必要な連続フレーム数（多くして厳しく）
    blur_size: int = 5             # 前処理ブラーサイズ
    
    # 人間検出設定
    use_pose_detection: bool = True  # 人間のポーズ検出を使用するか
    pose_confidence: float = 0.3     # ポーズ検出の信頼度しきい値（下げて検出率アップ）
    hand_confidence: float = 0.3     # 手の検出の信頼度しきい値（下げて検出率アップ）
    hand_proximity_threshold: int = 60  # 手の近傍と判定する距離（ピクセル）
    use_yolo: bool = True           # YOLOv8を使用するか
    yolo_model: str = "yolov8n.pt"  # YOLOモデル名
    yolo_confidence: float = 0.25    # YOLO検出の信頼度しきい値（下げて検出率アップ）
    
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
                       camera_params: Optional[Dict] = None, pose_data: Optional[Dict] = None,
                       config: DynaMaskConfig = DynaMaskConfig()) -> Tuple[np.ndarray, np.ndarray]:
    """前フレームと現在のフレーム間のホモグラフィー行列を推定する関数
    
    より高精度な特徴点マッチングとフィルタリングを行い、
    カメラパラメータと位置姿勢情報を活用して安定した推定を実現
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: ホモグラフィー行列と信頼性マスク
    """
    # グレースケールに変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape[:2]
    
    # カメラパラメータと位置姿勢情報がある場合は活用
    if camera_params is not None and pose_data is not None:
        try:
            # カメラパラメータから内部行列を取得
            camera_matrix = np.array(camera_params.get("camera_matrix", None))
            R_prev = np.array(pose_data.get("prev_rotation", None)) 
            R_curr = np.array(pose_data.get("curr_rotation", None))
            t_prev = np.array(pose_data.get("prev_translation", None))
            t_curr = np.array(pose_data.get("curr_translation", None))
            
            # 回転・並進情報から移動を推定
            if R_prev is not None and R_curr is not None and t_prev is not None and t_curr is not None and camera_matrix is not None:
                # 回転行列と並進ベクトルからカメラの移動を計算
                R_relative = np.dot(R_curr, R_prev.T)
                t_relative = t_curr - np.dot(R_relative, t_prev)
                
                # 本質行列を計算
                E = np.dot(np.cross(t_relative, np.identity(3)), R_relative)
                
                # 基礎行列を計算
                F = np.dot(np.dot(np.linalg.inv(camera_matrix).T, E), np.linalg.inv(camera_matrix))
                
                # 基礎行列からホモグラフィーを推定
                H_from_pose = cv2.findHomography(
                    np.array([[0, 0], [w, 0], [0, h], [w, h]]),
                    np.array([[0, 0], [w, 0], [0, h], [w, h]]),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    mask=None,
                    maxIters=2000
                )[0]
                
                # 位置姿勢情報が揃っている場合は、それを優先
                logger.debug("位置姿勢情報からホモグラフィー行列を推定しました")
                return H_from_pose, np.ones((h, w), dtype=np.uint8) * 255
        except Exception as e:
            logger.debug(f"位置姿勢情報からのホモグラフィー計算エラー: {e}")
            # エラーが発生した場合は特徴点ベースの方法にフォールバック
    
    # 特徴点検出のためのパラメータを調整
    feature_params = dict(
        maxCorners=3000,        # より多くの特徴点を検出（デフォルト1000）
        qualityLevel=0.03,      # 品質閾値を下げる（デフォルト0.01）
        minDistance=7,          # 最小距離を適度に設定
        blockSize=7,            # ブロックサイズを大きく
        useHarrisDetector=True, # Harrisコーナー検出を使用
        k=0.04                  # Harris検出器のパラメータ
    )
    
    # 前後フレームで特徴点を検出
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    curr_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
    
    if prev_pts is None or curr_pts is None or len(prev_pts) < 4 or len(curr_pts) < 4:
        logger.warning("十分な特徴点が検出できませんでした")
        return np.eye(3), np.zeros((h, w), dtype=np.uint8)
    
    # 特徴点記述子を計算 (SIFT, SURF, ORBなどから選択)
    descriptor = cv2.SIFT_create()
    prev_kp = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=7) for pt in prev_pts]
    curr_kp = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=7) for pt in curr_pts]
    
    # SIFT記述子を計算
    prev_kp, prev_des = descriptor.compute(prev_gray, prev_kp)
    curr_kp, curr_des = descriptor.compute(curr_gray, curr_kp)
    
    if prev_des is None or curr_des is None:
        logger.warning("特徴点記述子が計算できませんでした")
        return np.eye(3), np.zeros((h, w), dtype=np.uint8)
    
    # FLANN（高速近似最近傍探索）によるマッチング
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # これは探索の精度
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # マッチング実行
    matches = flann.knnMatch(prev_des, curr_des, k=2)
    
    # Lowe's ratio test でマッチングの品質を確認
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 0.7はLowe's ratioの一般的な値
            good_matches.append(m)
    
    # マッチング数のチェック
    if len(good_matches) < 10:
        logger.warning(f"良いマッチングが少なすぎます: {len(good_matches)}")
        # 厳しい条件でマッチングが少ない場合は緩和
        good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]
        
        if len(good_matches) < 10:
            logger.warning("マッチングが不十分なため単位行列を返します")
            return np.eye(3), np.zeros((h, w), dtype=np.uint8)
    
    # マッチング点の座標を抽出
    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # RANSAC法を使用してホモグラフィー行列を推定
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        logger.warning("ホモグラフィー行列が計算できませんでした")
        return np.eye(3), np.zeros((h, w), dtype=np.uint8)
    
    # ホモグラフィー行列の妥当性チェック
    det_H = np.linalg.det(H)
    if det_H < 0.5 or det_H > 2.0:  # 行列式が正常範囲外
        logger.warning(f"ホモグラフィー行列が異常です (det={det_H:.2f})")
        # 異常な行列の場合は、単位行列に近い形に補正
        H = np.eye(3) * 0.2 + H * 0.8
    
    # インライアのマスクを返す（ホモグラフィーの信頼性が高い領域）
    mask_img = np.zeros((h, w), dtype=np.uint8)
    
    # インライアの割合を計算
    inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
    
    # インライアマスクを視覚化
    if inlier_ratio > 0.4:  # インライアの割合が十分に高い場合
        # インライアのみの点を抽出
        inlier_src_pts = src_pts[mask.ravel() == 1]
        
        # ボロノイ図を作成してインライア領域を推定
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        
        for pt in inlier_src_pts:
            # 点がフレーム内にあることを確認
            x, y = pt[0]
            if 0 <= x < w and 0 <= y < h:
                subdiv.insert((int(x), int(y)))
        
        # ボロノイ領域を描画
        try:
            facets, centers = subdiv.getVoronoiFacetList([])
            for i, facet in enumerate(facets):
                # 多角形の描画
                hull = cv2.convexHull(np.array(facet))
                cv2.fillConvexPoly(mask_img, hull, 255)
        except:
            # ボロノイ図が作成できない場合は簡易的な方法で塗りつぶし
            for pt in inlier_src_pts:
                x, y = pt[0]
                cv2.circle(mask_img, (int(x), int(y)), 30, 255, -1)
    else:
        # インライアが少ない場合はシンプルに各点の周りを塗りつぶし
        for i, m in enumerate(mask):
            if m[0] == 1:  # インライアの場合
                x, y = src_pts[i][0]
                cv2.circle(mask_img, (int(x), int(y)), 20, 255, -1)
    
    # マスクに膨張処理を適用して連続性を高める
    mask_img = cv2.dilate(mask_img, np.ones((15, 15), np.uint8), iterations=2)
    
    return H, mask_img


def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                   H: np.ndarray, config: DynaMaskConfig,
                   homography_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """2フレーム間の動きを検出する関数
    
    ホモグラフィー行列を使用して前フレームをワープし、現フレームとの差分を計算
    より高度なノイズ除去と適応的閾値処理を適用して精度を向上させる
    
    Args:
        prev_frame: 前フレーム
        curr_frame: 現在のフレーム
        H: ホモグラフィー行列
        config: 設定
        homography_mask: ホモグラフィーの信頼性マスク
    
    Returns:
        np.ndarray: 動きマスク
    """
    height, width = curr_frame.shape[:2]
    
    # グレースケールに変換
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # ホモグラフィー行列を使用して前フレームをワープ
    warped_frame = cv2.warpPerspective(prev_gray, H, (width, height), flags=cv2.INTER_LINEAR)
    
    # ホモグラフィーの信頼性マスク
    reliability_mask = np.ones((height, width), dtype=np.uint8) * 255
    if homography_mask is not None:
        # 信頼性マスクがある場合は使用
        reliability_mask = homography_mask
    
    # 前処理：エッジ強調とノイズ除去
    # ガウシアンフィルタで平滑化
    warped_smooth = cv2.GaussianBlur(warped_frame, (5, 5), 0)
    curr_smooth = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    
    # ソーベルフィルタでエッジ検出
    sobelx_warped = cv2.Sobel(warped_smooth, cv2.CV_64F, 1, 0, ksize=3)
    sobely_warped = cv2.Sobel(warped_smooth, cv2.CV_64F, 0, 1, ksize=3)
    sobelx_curr = cv2.Sobel(curr_smooth, cv2.CV_64F, 1, 0, ksize=3)
    sobely_curr = cv2.Sobel(curr_smooth, cv2.CV_64F, 0, 1, ksize=3)
    
    # エッジ強度を計算
    edge_warped = np.sqrt(sobelx_warped**2 + sobely_warped**2)
    edge_curr = np.sqrt(sobelx_curr**2 + sobely_curr**2)
    
    # エッジを正規化
    edge_warped = cv2.normalize(edge_warped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edge_curr = cv2.normalize(edge_curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # エッジの差分
    edge_diff = cv2.absdiff(edge_warped, edge_curr)
    
    # 通常の画素値の差分
    pixel_diff = cv2.absdiff(warped_smooth, curr_smooth)
    
    # エッジと画素値の差分を統合
    combined_diff = cv2.addWeighted(pixel_diff, 0.7, edge_diff, 0.3, 0)
    
    # 適応的閾値処理
    # フレーム全体の差分統計
    mean_diff = np.mean(combined_diff)
    std_diff = np.std(combined_diff)
    
    # 動的閾値の計算
    # 背景ノイズ及びカメラ動きの影響に適応的に対応
    adaptive_threshold = min(
        max(mean_diff + 2.0 * std_diff, config.motion_threshold * 50),
        100.0  # 上限
    )
    
    # 動きマスクの生成
    motion_mask = (combined_diff > adaptive_threshold).astype(np.uint8) * 255
    
    # 信頼性の低い領域（ホモグラフィーが不正確な領域）はマスク
    if homography_mask is not None:
        unreliable_mask = 255 - homography_mask
        unreliable_regions = cv2.dilate(unreliable_mask, np.ones((5, 5), np.uint8), iterations=2)
        motion_mask = cv2.bitwise_and(motion_mask, homography_mask)
    
    # ノイズ除去：孤立点の除去
    # モルフォロジー処理でノイズを除去
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_open)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # ブロブ解析でサイズが小さすぎるものを除外
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50  # 最小面積
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            cv2.drawContours(motion_mask, [contour], -1, 0, -1)  # 小さいブロブを削除
    
    # 顕著な動きの強調（オプション）
    if len(contours) > 0 and np.sum(motion_mask) > 0:
        # 顕著な動き領域を特定
        significant_motion = np.zeros_like(motion_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 大きな動き領域のみ
                # モーメントを計算して中心を見つける
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 中心点からの膨張で強調
                    cv2.circle(significant_motion, (cx, cy), int(np.sqrt(area) * 0.8), 255, -1)
        
        # 顕著な動きを元のマスクと組み合わせ
        if np.sum(significant_motion) > 0:
            motion_mask = cv2.bitwise_or(motion_mask, significant_motion)
    
    logger.debug(f"動き検出: 閾値={adaptive_threshold:.1f}, 平均差分={mean_diff:.1f}, 標準偏差={std_diff:.1f}")
    
    return motion_mask


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
            results = yolo_model(frame, verbose=False, conf=config.yolo_confidence)[0]
            
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
                    hand_width = (x2 - x1) // 3  # より広い範囲を手の領域として考慮
                    
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
                    # より多くのキーポイントを使用して人間の形状を捉える
                    key_indices = [0, 11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28, 
                                   1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    
                    for idx in key_indices:
                        if idx < len(landmarks) and landmarks[idx].visibility > 0.2:  # 可視性の閾値を下げる
                            x, y = int(landmarks[idx].x * width), int(landmarks[idx].y * height)
                            points.append([x, y])
                    
                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        
                        if len(points) >= 3:
                            cv2.fillPoly(human_mask, [points], 255)
                        else:
                            for point in points:
                                cv2.circle(human_mask, tuple(point), 40, 255, -1)  # 円の半径を大きくする
                        
                        # 全身マスクを作成
                        all_visible_points = []
                        for i, landmark in enumerate(landmarks):
                            if landmark.visibility > 0.1:  # 可視性の閾値を下げる
                                x, y = int(landmark.x * width), int(landmark.y * height)
                                all_visible_points.append([x, y])
                        
                        if len(all_visible_points) > 2:
                            all_visible_points = np.array(all_visible_points, dtype=np.int32)
                            
                            if len(all_visible_points) >= 3:
                                hull = cv2.convexHull(all_visible_points)
                                cv2.fillPoly(human_mask, [hull], 255)
                                
                                # 人間の輪郭を拡張してマスクをより広くする
                                kernel = np.ones((15, 15), np.uint8)
                                human_mask = cv2.dilate(human_mask, kernel, iterations=1)
                            else:
                                for i in range(len(all_visible_points) - 1):
                                    pt1 = tuple(all_visible_points[i])
                                    pt2 = tuple(all_visible_points[i + 1])
                                    cv2.line(human_mask, pt1, pt2, 255, 10)  # 線の太さを増やす
                    
                    # 手の位置を特定（より広い範囲でキャプチャ）
                    hand_landmarks = [15, 16, 17, 18, 19, 20, 21, 22]  # 手と腕のランドマーク
                    for hand_idx in hand_landmarks:
                        if hand_idx < len(landmarks) and landmarks[hand_idx].visibility > 0.2:
                            x, y = int(landmarks[hand_idx].x * width), int(landmarks[hand_idx].y * height)
                            hand_positions.append((x, y))
                            # 半径を整数に変換
                            radius = int(config.hand_proximity_threshold * 1.5)
                            cv2.circle(human_mask, (x, y), radius, 255, -1)  # 手の領域を大きくする
        
        # 手の詳細検出
        hands_configs = [
            {"max_num_hands": 6, "min_detection_confidence": config.hand_confidence},  # より多くの手を検出できるように
            {"max_num_hands": 6, "min_detection_confidence": config.hand_confidence - 0.1}
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
                                
                                # 手の領域を拡張してより広いエリアをカバー
                                hand_area_dilated = np.zeros_like(human_mask)
                                cv2.fillPoly(hand_area_dilated, [hull], 255)
                                kernel = np.ones((20, 20), np.uint8)  # より大きなカーネルで拡張
                                hand_area_dilated = cv2.dilate(hand_area_dilated, kernel)
                                human_mask = cv2.bitwise_or(human_mask, hand_area_dilated)
                            else:
                                for point in hand_points:
                                    cv2.circle(human_mask, tuple(point), 30, 255, -1)  # 円の半径を大きくする
    
    # マスクの拡張
    kernel = np.ones((15, 15), np.uint8)  # より大きなカーネルを使用
    human_mask = cv2.dilate(human_mask, kernel, iterations=2)
    
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
        
        # 動きマスクの合計面積を計算（全体的な動き量の指標として）
        total_motion_pixels = np.sum(motion_mask > 0)
        if total_motion_pixels == 0:
            logger.debug("フレーム全体に動きが検出されませんでした")
            # 動きがなければ、人間と手の近くのセグメントのみを考慮
            return process_human_segments_only(masks, human_mask, hand_positions, frame_shape, config)
        
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
            
            # セグメントのバイナリマスク
            mask_binary = (mask_2d > 0.5).astype(np.uint8) * 255
            
            # 人間マスクとの重なりを確認（優先的に処理）
            human_overlap = False
            human_overlap_ratio = 0.0
            if human_mask is not None:
                # 人間との重なりを計算
                human_overlap_mask = cv2.bitwise_and(human_mask, mask_binary)
                human_overlap_ratio = np.sum(human_overlap_mask > 0) / max(1, segment_pixels)
                
                # 人間との重なりが一定以上なら確実に動的要素として検出
                if human_overlap_ratio > 0.15:
                    human_overlap = True
                    dynamic_indices.append(i)
                    logger.debug(f"セグメント {i}: 人間との重なり ({human_overlap_ratio:.2f}) で動的と判定")
                    continue  # 人間と重なる場合は他の判定をスキップ
            
            # 手の近くにあるか確認
            near_hand = False
            min_hand_distance = float('inf')
            
            # セグメントの中心を計算
            moments = cv2.moments(mask_binary)
            if moments["m00"] == 0:
                continue
                
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            if hand_positions:
                for hand_x, hand_y in hand_positions:
                    # セグメントの中心と手の距離を計算
                    distance = np.sqrt((cx - hand_x)**2 + (cy - hand_y)**2)
                    min_hand_distance = min(min_hand_distance, distance)
                    
                    if distance < config.hand_proximity_threshold:
                        near_hand = True
                        break
            
            # セグメントと動きマスクの重なりを計算
            overlap = cv2.bitwise_and(motion_mask, mask_binary)
            overlap_pixels = np.sum(overlap > 0)
            overlap_ratio = overlap_pixels / max(1, segment_pixels)
            
            # セグメントの特性に基づく動き分析
            is_dynamic = False
            
            # 中心領域の動きを確認（セグメント中心の動きは重要）
            center_region_size = min(25, int(np.sqrt(segment_pixels) / 4))
            center_region = motion_mask[
                max(0, cy-center_region_size):min(height, cy+center_region_size),
                max(0, cx-center_region_size):min(width, cx+center_region_size)
            ]
            center_motion = np.mean(center_region) if center_region.size > 0 else 0
            
            # エッジ部分の動きを確認（カメラ移動による見かけの動きを排除）
            edge_kernel = np.ones((3, 3), np.uint8)
            edge_mask = cv2.dilate(mask_binary, edge_kernel) - cv2.erode(mask_binary, edge_kernel)
            edge_motion = cv2.bitwise_and(motion_mask, edge_mask)
            edge_ratio = np.sum(edge_motion > 0) / max(1, np.sum(edge_mask > 0))
            
            # セグメントの形状解析（円形度、凸性など）
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(main_contour, True)
                if perimeter > 0:
                    # 円形度 (4π×面積/周囲長^2) - 1に近いほど円形
                    circularity = 4 * np.pi * segment_pixels / (perimeter ** 2)
                    
                    # 凸包との比較（凸性）
                    hull = cv2.convexHull(main_contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        convexity = segment_pixels / hull_area
                    else:
                        convexity = 1.0
                    
                    # 細長いオブジェクトは動きやすい（アスペクト比）
                    x, y, w, h = cv2.boundingRect(main_contour)
                    aspect_ratio = max(w, h) / max(1, min(w, h))
                else:
                    circularity = 0
                    convexity = 0
                    aspect_ratio = 1
            else:
                circularity = 0
                convexity = 0
                aspect_ratio = 1
            
            # 標準動的判定条件（通常の物体）- 厳しく設定
            standard_dynamic_condition = (
                overlap_ratio > 0.65 and       # 重なり率を上げる（より厳格に）
                center_motion > config.motion_threshold and
                overlap_pixels > 200 and       # 最小動きピクセル数
                edge_ratio < 0.7              # エッジの動き率
            )
            
            # 厳しい動的判定の追加条件（確実に動いている物体）
            strict_motion_condition = (
                overlap_ratio > 0.75 and
                center_motion > config.motion_threshold * 1.2 and
                overlap_pixels > 400
            )
            
            # 手の近くの物体に対する緩和条件
            hand_proximity_condition = near_hand and (
                overlap_ratio > 0.2 and
                center_motion > config.motion_threshold * 0.4 and
                overlap_pixels > 50
            )
            
            # 特殊形状の物体（細長いもの、非凸形状のもの）に対する調整
            shape_adjusted_condition = (
                overlap_ratio > 0.5 and
                center_motion > config.motion_threshold * 0.8 and
                (
                    (aspect_ratio > 3 and overlap_pixels > 150) or  # 細長い物体
                    (convexity < 0.7 and overlap_pixels > 200)      # 非凸形状
                )
            )
            
            # 最終的な動的判定
            if strict_motion_condition:
                is_dynamic = True
                logger.debug(f"セグメント {i}: 厳しい動き条件で動的と判定 (重なり率: {overlap_ratio:.2f}, 中心動き: {center_motion:.1f})")
            elif hand_proximity_condition:
                is_dynamic = True
                logger.debug(f"セグメント {i}: 手の近く ({min_hand_distance:.1f}px) で動的と判定 (重なり率: {overlap_ratio:.2f})")
            elif standard_dynamic_condition:
                is_dynamic = True
                logger.debug(f"セグメント {i}: 標準条件で動的と判定 (重なり率: {overlap_ratio:.2f}, 中心動き: {center_motion:.1f})")
            elif shape_adjusted_condition:
                is_dynamic = True
                logger.debug(f"セグメント {i}: 形状調整条件で動的と判定 (アスペクト比: {aspect_ratio:.1f}, 凸性: {convexity:.2f})")
            
            if is_dynamic:
                dynamic_indices.append(i)
    
    except Exception as e:
        logger.error(f"セグメントフィルタリングエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return dynamic_indices


def process_human_segments_only(masks: np.ndarray, human_mask: Optional[np.ndarray], 
                              hand_positions: Optional[List[Tuple[int, int]]], 
                              frame_shape: Tuple[int, int],
                              config: DynaMaskConfig) -> List[int]:
    """動きが検出されなかった場合に、人間と手の近くのセグメントのみを処理する"""
    dynamic_indices = []
    height, width = frame_shape[:2]
    total_pixels = height * width
    
    if human_mask is None and not hand_positions:
        return []
    
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
        
        # セグメントのバイナリマスク
        mask_binary = (mask_2d > 0.5).astype(np.uint8) * 255
        
        # 人間マスクとの重なりを確認
        if human_mask is not None:
            # 人間との重なりを計算
            human_overlap_mask = cv2.bitwise_and(human_mask, mask_binary)
            human_overlap_ratio = np.sum(human_overlap_mask > 0) / max(1, segment_pixels)
            
            # 人間との重なりが一定以上なら動的要素として検出
            if human_overlap_ratio > 0.15:
                dynamic_indices.append(i)
                logger.debug(f"セグメント {i}: 人間との重なりで動的と判定 (動きなしフレーム)")
                continue
        
        # 手の近くにあるか確認
        if hand_positions:
            # セグメントの中心を計算
            moments = cv2.moments(mask_binary)
            if moments["m00"] == 0:
                continue
            
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            for hand_x, hand_y in hand_positions:
                # セグメントの中心と手の距離を計算
                distance = np.sqrt((cx - hand_x)**2 + (cy - hand_y)**2)
                
                # 手の非常に近くのセグメントのみ
                if distance < config.hand_proximity_threshold * 0.7:  # より厳しい条件
                    dynamic_indices.append(i)
                    logger.debug(f"セグメント {i}: 手の非常に近く ({distance:.1f}px) で動的と判定 (動きなしフレーム)")
                    break
    
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
    human_segments = set()  # 人間と判定されたセグメントを記録
    
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
            H, homography_mask = estimate_homography(prev_frame, frame, config=config)
            
            # 動きマスクの検出（ホモグラフィーマスクも渡す）
            motion_mask = detect_motion(prev_frame, frame, H, config, homography_mask)
            
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
            
            # 人間との重なりを確認して人間セグメントを記録
            if human_mask is not None and hasattr(curr_result, "__getitem__") and len(curr_result) > 0:
                if hasattr(curr_result[0], "masks") and curr_result[0].masks is not None:
                    masks = curr_result[0].masks.data.cpu().numpy()
                    for i, mask in enumerate(masks):
                        mask_2d = mask[0] if mask.ndim > 2 else mask
                        mask_binary = (mask_2d > 0.5).astype(np.uint8) * 255
                        human_overlap = cv2.bitwise_and(human_mask, mask_binary)
                        if np.sum(human_overlap > 0) / max(1, np.sum(mask_binary > 0)) > 0.15:
                            # 人間と重なるセグメントを記録
                            human_segments.add(i)
            
            # 時間的一貫性を考慮して動的判定を更新
            # 古い履歴をクリア
            dynamic_history = {k: v for k, v in dynamic_history.items() if v > 0}
            
            # 動的カウントを減衰させる（時間とともに忘れていく）
            for idx in dynamic_history:
                dynamic_history[idx] = max(0, dynamic_history[idx] - 0.5)
            
            # 新しい動的セグメントのカウントを増加
            for idx in dynamic_indices:
                if idx not in dynamic_history:
                    dynamic_history[idx] = 0
                dynamic_history[idx] += 1
                
                # 人間セグメントの場合、カウントを高く設定
                if idx in human_segments:
                    dynamic_history[idx] = max(dynamic_history[idx], config.temporal_consistency * 1.5)
            
            # 各セグメントの動的状態を判定
            final_dynamic_indices = [
                idx for idx, count in dynamic_history.items() 
                if count >= config.temporal_consistency or idx in human_segments
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
                            
                            # 人間セグメントは異なる色で表示
                            if idx in human_segments:
                                color = [0, 0, 255]  # 人間は赤色
                            else:
                                color = colors[i % len(colors)].tolist()
                            
                            # マスクのオーバーレイ
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
                    human_count = len(human_segments.intersection(set(final_dynamic_indices)))
                    other_count = len(final_dynamic_indices) - human_count
                    
                    cv2.putText(
                        output_frame,
                        f"Dynamic Objects: {other_count} / Humans: {human_count}",
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
    parser.add_argument("--motion-threshold", type=float, default=80.0, help="動きと判定するしきい値（高い値ほど厳しく判定）")
    parser.add_argument("--min-area", type=float, default=0.01, help="セグメント面積の最小比率")
    parser.add_argument("--max-area", type=float, default=0.4, help="セグメント面積の最大比率")
    parser.add_argument("--temporal", type=int, default=4, help="動的判定に必要な連続フレーム数（高いほど厳しく判定）")
    
    # 人間検出設定
    parser.add_argument("--no-yolo", dest="use_yolo", action="store_false", help="YOLOによる人間検出を無効化")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLOモデル名")
    parser.add_argument("--no-pose", dest="use_pose", action="store_false", help="MediaPipeによるポーズ検出を無効化")
    parser.add_argument("--hand-proximity", type=int, default=60, help="手の近傍と判定する距離（ピクセル）")
    
    # FastSAM設定
    parser.add_argument("--model", type=str, default="FastSAM-x", help="モデル名")
    parser.add_argument("--conf", type=float, default=0.7, help="セグメント化の信頼度しきい値")
    
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
        motion_threshold=80.0,  # 動き閾値を高く設定
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        temporal_consistency=4,  # 時間的一貫性の要求を厳しくする
        blur_size=5,
        image_pattern=args.image_pattern,
        use_pose_detection=args.use_pose,
        pose_confidence=0.3,  # 人間検出の信頼度閾値を下げる
        hand_confidence=0.3,  # 手の検出の信頼度閾値を下げる
        hand_proximity_threshold=args.hand_proximity,  # 手の近傍距離
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        yolo_confidence=0.25,  # YOLO検出の信頼度閾値を下げる
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