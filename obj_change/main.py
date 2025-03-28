import os
import sys
import cv2
import numpy as np
import torch
import time
import json
from tqdm import tqdm
import argparse
from pathlib import Path

# 深度推定のためのDepth Anything V2
from depth_estimator import DepthEstimator

# 点群処理のためのクラス
from point_cloud_processor import PointCloudProcessor

# セグメンテーションのためのFastSAM
from segmentation import SegmentationModel

# 動的物体検出のためのクラス
from dynamic_object_detector import DynamicObjectDetector

def load_camera_params(params_file):
    """
    カメラパラメータをJSONファイルから読み込む
    
    Args:
        params_file: カメラパラメータのJSONファイルパス
        
    Returns:
        カメラパラメータの辞書（カメラIDをキーとして）
    """
    try:
        with open(params_file, 'r') as f:
            camera_params = json.load(f)
        return camera_params
    except Exception as e:
        print(f"カメラパラメータの読み込みに失敗しました: {e}")
        sys.exit(1)

def load_images(image_paths):
    """
    画像のリストを読み込む
    
    Args:
        image_paths: 画像ファイルパスのリスト
        
    Returns:
        画像のリスト（NumPy配列）
    """
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"警告: 画像ファイルが見つかりません: {path}")
            continue
        
        img = cv2.imread(path)
        if img is None:
            print(f"警告: 画像の読み込みに失敗しました: {path}")
            continue
            
        images.append(img)
        
    return images

def visualize_results(latest_image, past_image, masked_image, dynamic_mask, output_dir):
    """
    結果を可視化して保存する
    
    Args:
        latest_image: 最新画像
        past_image: 過去画像（元画像）
        masked_image: マスク適用後の過去画像
        dynamic_mask: 動的物体マスク
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # マスク画像を可視化（白黒の2値画像を疑似カラー化）
    color_mask = cv2.applyColorMap((dynamic_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 透明度付きのオーバーレイ画像を作成
    alpha = 0.5
    overlay = cv2.addWeighted(past_image, 1-alpha, color_mask, alpha, 0)
    
    # 比較表示用に画像を並べる
    h, w = latest_image.shape[:2]
    h_new, w_new = h, w * 4  # 4つの画像を横に並べる
    
    comparison = np.zeros((h_new, w_new, 3), dtype=np.uint8)
    comparison[:, :w] = latest_image
    comparison[:, w:w*2] = past_image
    comparison[:, w*2:w*3] = masked_image
    comparison[:, w*3:] = overlay
    
    # ラベルを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Latest Image", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Past Image", (w+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Masked Image", (w*2+10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Dynamic Mask", (w*3+10, 30), font, 1, (0, 255, 0), 2)
    
    # 結果を保存
    timestamp = int(time.time())
    cv2.imwrite(os.path.join(output_dir, f"comparison_{timestamp}.jpg"), comparison)
    cv2.imwrite(os.path.join(output_dir, f"latest_{timestamp}.jpg"), latest_image)
    cv2.imwrite(os.path.join(output_dir, f"past_{timestamp}.jpg"), past_image)
    cv2.imwrite(os.path.join(output_dir, f"masked_{timestamp}.jpg"), masked_image)
    cv2.imwrite(os.path.join(output_dir, f"mask_{timestamp}.jpg"), dynamic_mask * 255)
    
    print(f"結果を保存しました: {output_dir}")
    
    return comparison

def process_image_pair(latest_image, past_image, latest_pose, past_pose, 
                      camera_intrinsics, depth_model, segmentation_model, 
                      point_cloud_processor, dynamic_detector,
                      distance_threshold=0.2, dynamic_threshold=0.5,
                      visualize=True, output_dir="output"):
    """
    1組の最新画像と過去画像のペアを処理する
    
    Args:
        latest_image: 最新画像（NumPy配列）
        past_image: 過去画像（NumPy配列）
        latest_pose: 最新画像のカメラ姿勢（4x4変換行列）
        past_pose: 過去画像のカメラ姿勢（4x4変換行列）
        camera_intrinsics: カメラの内部パラメータ（fx, fy, cx, cy）
        depth_model: 深度推定モデル
        segmentation_model: セグメンテーションモデル
        point_cloud_processor: 点群処理クラス
        dynamic_detector: 動的物体検出クラス
        distance_threshold: 動的物体と判定する距離閾値（メートル）
        dynamic_threshold: セグメントが動的と判定する閾値（セグメント内の動的ピクセル割合）
        visualize: 結果を可視化するかどうか
        output_dir: 出力ディレクトリ
        
    Returns:
        マスク適用後の過去画像、動的物体マスク、（オプション）比較画像
    """
    print("\n--- 画像ペアの処理を開始 ---")
    
    # 1. 深度推定
    print("1. 深度推定中...")
    latest_depth = depth_model.estimate_depth(latest_image)
    past_depth = depth_model.estimate_depth(past_image)
    
    # 2. 点群生成
    print("2. 点群生成中...")
    latest_points = depth_model.generate_point_cloud(latest_image, latest_depth, camera_intrinsics)
    past_points = depth_model.generate_point_cloud(past_image, past_depth, camera_intrinsics)
    
    # 3. 過去画像の点群を最新画像の視点に変換
    print("3. 点群座標変換中...")
    # 過去視点から最新視点への変換行列を計算 (past → world → latest)
    past_to_world = past_pose
    world_to_latest = np.linalg.inv(latest_pose)
    past_to_latest = np.dot(world_to_latest, past_to_world)
    
    # 点群変換を適用
    past_points_transformed = point_cloud_processor.transform_point_cloud(past_points, past_to_latest)
    
    # 4. 点群比較（最近傍探索で動的点を検出）
    print("4. 点群比較（動的物体検出）中...")
    # 最新画像の点群からFaissインデックスを構築
    latest_index = point_cloud_processor.build_faiss_index(latest_points)
    
    # 変換後の過去点群と最近傍比較
    dynamic_indices, distances = point_cloud_processor.find_nearest_neighbors(
        past_points_transformed, latest_index, distance_threshold=distance_threshold
    )
    
    # 5. 過去画像のセグメンテーション
    print("5. 過去画像のセグメンテーション中...")
    past_segments = segmentation_model.segment_image(past_image)
    
    # 6. 動的点群の投影＆動的セグメントの検出
    print("6. 動的セグメント検出中...")
    # 動的点群を過去画像の平面に投影
    image_points, valid_mask = dynamic_detector.project_points_to_image(
        past_points_transformed, camera_intrinsics, past_image.shape
    )
    
    # 動的セグメントを検出
    dynamic_segment_indices, segment_ratios = dynamic_detector.detect_dynamic_segments(
        dynamic_indices, image_points, valid_mask, past_segments
    )
    
    print(f"動的セグメント検出結果: {len(dynamic_segment_indices)}/{len(past_segments)}個のセグメントが動的と判定")
    for i, idx in enumerate(dynamic_segment_indices):
        if i < 10:  # 最初の10個のみ表示
            print(f"  セグメント {idx}: 動的ピクセル比率 {segment_ratios[idx]:.2f}")
    
    # 7. 動的マスクの生成
    print("7. 動的マスク生成中...")
    dynamic_mask = dynamic_detector.create_dynamic_mask(
        dynamic_segment_indices, past_segments, past_image.shape
    )
    
    # 8. マスク適用
    print("8. マスク適用中...")
    masked_past_image = dynamic_detector.apply_mask_to_image(past_image, dynamic_mask)
    
    # 9. 結果の可視化（オプション）
    comparison = None
    if visualize:
        print("9. 結果の可視化...")
        comparison = visualize_results(
            latest_image, past_image, masked_past_image, dynamic_mask, output_dir
        )
    
    print("--- 処理完了 ---")
    
    return masked_past_image, dynamic_mask, comparison

def main():
    """
    メイン関数 - 動的物体検出とマスク処理のワークフローを実行
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="動的物体検出とマスク処理システム")
    parser.add_argument("--latest", required=True, help="最新画像のパス")
    parser.add_argument("--past", required=True, help="過去画像のパス")
    parser.add_argument("--poses", required=True, help="カメラ姿勢情報のJSONファイル")
    parser.add_argument("--intrinsics", required=True, help="カメラ内部パラメータのJSONファイル")
    parser.add_argument("--output", default="output", help="出力ディレクトリ")
    parser.add_argument("--dist_threshold", type=float, default=0.2, help="動的点判定の距離閾値（メートル）")
    parser.add_argument("--dyn_threshold", type=float, default=0.5, help="動的セグメント判定の閾値")
    args = parser.parse_args()
    
    print("動的物体検出とマスク処理システムを開始します...")
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # モデルの初期化
    depth_model = DepthEstimator()
    segmentation_model = SegmentationModel()
    point_cloud_processor = PointCloudProcessor()
    dynamic_detector = DynamicObjectDetector(dynamic_threshold=args.dyn_threshold)
    
    # カメラパラメータの読み込み
    try:
        # カメラ姿勢（外部パラメータ）
        with open(args.poses, 'r') as f:
            poses = json.load(f)
            
        # カメラの内部パラメータ
        with open(args.intrinsics, 'r') as f:
            intrinsics = json.load(f)
            
        # 内部パラメータの抽出（fx, fy, cx, cy）
        camera_intrinsics = (
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"]
        )
    except Exception as e:
        print(f"カメラパラメータの読み込みに失敗しました: {e}")
        sys.exit(1)
    
    # 画像の読み込み
    latest_image = cv2.imread(args.latest)
    past_image = cv2.imread(args.past)
    
    if latest_image is None or past_image is None:
        print("画像の読み込みに失敗しました。ファイルパスを確認してください。")
        sys.exit(1)
        
    # カメラ姿勢の抽出
    latest_id = os.path.basename(args.latest).split('.')[0]
    past_id = os.path.basename(args.past).split('.')[0]
    
    if latest_id not in poses or past_id not in poses:
        print(f"カメラ姿勢情報が見つかりません。ID: {latest_id} または {past_id}")
        sys.exit(1)
        
    latest_pose = np.array(poses[latest_id])
    past_pose = np.array(poses[past_id])
    
    # 画像ペア処理
    masked_past_image, dynamic_mask, comparison = process_image_pair(
        latest_image, past_image, latest_pose, past_pose, camera_intrinsics,
        depth_model, segmentation_model, point_cloud_processor, dynamic_detector,
        distance_threshold=args.dist_threshold,
        dynamic_threshold=args.dyn_threshold,
        output_dir=args.output
    )
    
    # 結果を表示（GUIがある場合）
    if comparison is not None and 'DISPLAY' in os.environ:
        cv2.imshow("Results", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"処理が完了しました。結果は {args.output} に保存されています。")

if __name__ == "__main__":
    main()
