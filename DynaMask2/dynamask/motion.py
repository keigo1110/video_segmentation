import cv2
import numpy as np
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple

from .utils import qvec_to_rotmat # ヘルパー関数をインポート

if TYPE_CHECKING:
    from .config import DynaMaskConfig

logger = logging.getLogger(__name__)

# 定数 (将来的に設定ファイルから読み込むことも検討)
DEFAULT_DEPTH_FOR_WARPING = 10.0 # COLMAPワーピングで使用する固定深度

def detect_motion_optical_flow(
    config: 'DynaMaskConfig',
    curr_frame: np.ndarray,
    prev_frame: np.ndarray,
    curr_frame_info: Optional[Dict[str, Any]] = None,
    prev_frame_info: Optional[Dict[str, Any]] = None,
    camera_params: Optional[Dict[str, Any]] = None,
    prev_hand_positions: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """フレーム間の動きを検出する。COLMAPによる自己運動分離と領域適応閾値処理をサポート。

    Args:
        config: 設定オブジェクト。
        curr_frame: 現在のフレーム (BGR)。
        prev_frame: 前のフレーム (BGR)。
        curr_frame_info: 現在フレームのCOLMAP情報 (images.txtのエントリ)。
        prev_frame_info: 前フレームのCOLMAP情報。
        camera_params: カメラの内部パラメータ (cameras.txtのエントリ)。
        prev_hand_positions: 前フレームで検出された手の中心座標リスト [(x1, y1), (x2, y2), ...] 。

    Returns:
        np.ndarray: 動きマスク (前景が255、背景が0のバイナリマスク)。
    """
    height, width = curr_frame.shape[:2]

    # グレースケールに変換
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

    # ノイズ除去
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    initial_motion_mask = np.zeros_like(curr_gray, dtype=np.uint8)
    intensity_map = None # 閾値処理対象のマップ (差分 or フロー強度)
    colmap_separation = False # COLMAPベースの分離を試みるか

    # --- Step 1: 自己運動分離方法の決定 ---
    can_try_colmap = (
        config.use_colmap_egomotion and
        curr_frame_info is not None and
        prev_frame_info is not None and
        camera_params is not None and
        camera_params.get('model') == 'PINHOLE' and # 簡単のためPINHOLEモデルのみ対応
        len(camera_params.get('params', [])) >= 4 # fx, fy, cx, cy が必要
    )

    if can_try_colmap:
        try:
            # --- Step 2: COLMAPベースの自己運動分離 --- 
            logger.debug(f"フレーム {curr_frame_info.get('name', '')} でCOLMAPベースの自己運動分離を試行")
            # カメラパラメータとポーズを取得
            fx, fy, cx, cy = camera_params['params'][:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K_inv = np.linalg.inv(K)

            q_prev = prev_frame_info['qvec']
            t_prev = prev_frame_info['tvec']
            R_prev = qvec_to_rotmat(q_prev)
            P_prev_world_to_cam = np.hstack((R_prev, t_prev.reshape(-1, 1)))

            q_curr = curr_frame_info['qvec']
            t_curr = curr_frame_info['tvec']
            R_curr = qvec_to_rotmat(q_curr)
            P_curr_world_to_cam = np.hstack((R_curr, t_curr.reshape(-1, 1)))

            # 現在->前フレームへの変換行列 (カメラ座標系)
            # P_prev = T_prev_world * P_world
            # P_curr = T_curr_world * P_world => P_world = T_curr_world_inv * P_curr
            # P_prev = T_prev_world * T_curr_world_inv * P_curr
            # World -> Cam の場合: T_cam_world = [R | t], T_world_cam = [R.T | -R.T @ t]
            R_curr_inv = R_curr.T
            t_curr_inv = -R_curr_inv @ t_curr.reshape(-1, 1)
            T_curr_world_to_cam_inv = np.vstack((np.hstack((R_curr_inv, t_curr_inv)), [0, 0, 0, 1]))

            T_prev_world_to_cam_hom = np.vstack((P_prev_world_to_cam, [0, 0, 0, 1]))

            T_curr_to_prev_cam = T_prev_world_to_cam_hom @ T_curr_world_to_cam_inv
            R_c2p = T_curr_to_prev_cam[:3, :3]
            t_c2p = T_curr_to_prev_cam[:3, 3]

            # ワーピングマップの計算
            map1 = np.zeros_like(curr_gray, dtype=np.float32)
            map2 = np.zeros_like(curr_gray, dtype=np.float32)
            coords = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
            u_coords = coords[..., 1]
            v_coords = coords[..., 0]

            # 逆投影 (u, v) -> 正規化画像座標 (x', y')
            pts_2d_hom = np.dstack((u_coords, v_coords, np.ones_like(u_coords)))
            pts_norm_cam = (K_inv @ pts_2d_hom.reshape(-1, 3).T).T.reshape(height, width, 3)

            # 深度を仮定して3D点へ (現在のカメラ座標系)
            depth = DEFAULT_DEPTH_FOR_WARPING
            pts_3d_curr_cam = pts_norm_cam * depth

            # 前フレームのカメラ座標系へ変換
            pts_3d_prev_cam = (R_c2p @ pts_3d_curr_cam.reshape(-1, 3).T + t_c2p.reshape(-1, 1)).T

            # 再投影 (前フレーム画像座標へ)
            pts_2d_prev_hom = (K @ pts_3d_prev_cam.reshape(-1, 3).T).T
            valid_depth_mask = pts_2d_prev_hom[:, 2] > 1e-5 # ゼロ割防止

            pts_2d_prev = np.zeros((height * width, 2), dtype=np.float32)
            pts_2d_prev[valid_depth_mask, 0] = pts_2d_prev_hom[valid_depth_mask, 0] / pts_2d_prev_hom[valid_depth_mask, 2]
            pts_2d_prev[valid_depth_mask, 1] = pts_2d_prev_hom[valid_depth_mask, 1] / pts_2d_prev_hom[valid_depth_mask, 2]

            map1 = pts_2d_prev[:, 0].reshape(height, width)
            map2 = pts_2d_prev[:, 1].reshape(height, width)

            # ワーピング実行
            warped_prev_gray = cv2.remap(prev_gray, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 差分計算
            intensity_map = cv2.absdiff(curr_gray, warped_prev_gray)
            colmap_separation = True
            logger.debug("COLMAPワーピングによる差分画像を計算しました。")

        except Exception as e:
            logger.warning(f"COLMAPベースの自己運動分離中にエラー: {e}。オプティカルフローにフォールバックします。", exc_info=False)
            colmap_separation = False
            intensity_map = None

    # --- Step 3: オプティカルフローベースの処理 (if not colmap_separation) ---
    if not colmap_separation:
        logger.debug("オプティカルフローベースの動き検出を実行します。")
        # 既存のフロー計算ロジック (縮小版は廃止、直接計算)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=config.optical_flow_winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0
        )

        # フローから動きベクトルの大きさ（マグニチュード）を計算
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # フロー補正 (中央値ベース)
        if config.fallback_to_flow_compensation:
            u_median = np.median(flow[..., 0])
            v_median = np.median(flow[..., 1])
            flow_compensated_x = flow[..., 0] - u_median
            flow_compensated_y = flow[..., 1] - v_median
            mag, _ = cv2.cartToPolar(flow_compensated_x, flow_compensated_y)
        
        intensity_map = mag

    # --- Step 4: 領域適応閾値処理 --- 
    if intensity_map is None:
        logger.error("強度マップが生成されていません。動きマスクは空になります。")
        return np.zeros_like(curr_gray, dtype=np.uint8)
    
    if config.use_region_adaptive_threshold and prev_hand_positions is not None and len(prev_hand_positions) > 0:
        logger.debug("領域適応閾値処理を実行します。")
        # 作業領域マスク生成 (前フレームの手の位置基準)
        work_region_mask = np.zeros_like(curr_gray, dtype=np.uint8)
        for (hx, hy) in prev_hand_positions:
            cv2.circle(work_region_mask, (int(hx), int(hy)), config.hand_region_radius, 255, -1)
        background_mask = cv2.bitwise_not(work_region_mask)

        # 閾値の計算
        mean_intensity = np.mean(intensity_map)
        std_intensity = np.std(intensity_map)
        
        # 背景閾値 T_bg (フロー/差分共通で適応的閾値を使用)
        # config.motion_threshold は適応閾値のスケール調整に使用する (要調整)
        # adaptive_scale = config.motion_threshold / 100.0 * 2.5 + 1.5 # 元のスケール計算例
        # 背景領域の閾値は少し高めに設定することが多い
        adaptive_scale_bg = config.motion_threshold / 50.0 # 例: motion_threshold=80 -> 1.6
        T_bg = mean_intensity + std_intensity * adaptive_scale_bg
        if T_bg < 1: T_bg = 1 # 最低閾値

        # 作業領域閾値 T_work
        T_work = T_bg * config.hand_region_threshold_factor
        if T_work < 1: T_work = 1 # 最低閾値

        logger.debug(f"閾値: 背景={T_bg:.2f}, 作業領域={T_work:.2f}")

        # 領域別閾値処理
        initial_motion_mask[ (work_region_mask > 0) & (intensity_map > T_work) ] = 255
        initial_motion_mask[ (background_mask > 0) & (intensity_map > T_bg) ] = 255

    else:
        # 通常の適応閾値 (領域適応OFF または 手が検出されていない場合)
        logger.debug("通常の適応閾値処理を実行します。")
        mean_intensity = np.mean(intensity_map)
        std_intensity = np.std(intensity_map)
        adaptive_scale = config.motion_threshold / 50.0 # 背景と同じスケールを使う
        threshold = mean_intensity + std_intensity * adaptive_scale
        if threshold < 1: threshold = 1 # 最低閾値
        logger.debug(f"閾値: {threshold:.2f}")
        initial_motion_mask = (intensity_map > threshold).astype(np.uint8) * 255

    # --- Step 5: 後処理 --- 
    # 形態素演算 (ノイズ除去と穴埋め)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    motion_mask = cv2.morphologyEx(initial_motion_mask, cv2.MORPH_OPEN, kernel_open)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_close)

    # 小さな連結成分を除去
    min_motion_pixels = config.min_motion_pixels # 設定から取得
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
    for i in range(1, num_labels):  # 0はバックグラウンド
        if stats[i, cv2.CC_STAT_AREA] < min_motion_pixels:
            motion_mask[labels == i] = 0

    # --- Step 6: 戻り値 --- 
    logger.debug("動きマスク生成完了")
    return motion_mask
