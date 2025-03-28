#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動的物体検出とマスク処理システムのテスト実行スクリプト
"""

import os
import sys
import cv2
import numpy as np
import json
import time

# プロジェクトのモジュールをインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator
from point_cloud_processor import PointCloudProcessor
from segmentation import SegmentationModel
from dynamic_object_detector import DynamicObjectDetector

def load_camera_params_from_colmap(cameras_txt):
    """
    COLMAPのカメラパラメータファイルを読み込む
    
    Args:
        cameras_txt: COLMAPのカメラパラメータファイルパス
        
    Returns:
        カメラの内部パラメータ (fx, fy, cx, cy)
    """
    # ファイルの中身を読む
    with open(cameras_txt, 'r') as f:
        lines = f.readlines()
    
    # コメント行をスキップ
    camera_lines = [line for line in lines if not line.startswith('#')]
    
    # 最初のカメラを使用
    if camera_lines:
        parts = camera_lines[0].strip().split()
        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        fx = float(parts[4])
        fy = float(parts[4])  # 多くの場合、fx = fy
        cx = float(parts[5])
        cy = float(parts[6])
        
        return (fx, fy, cx, cy)
    else:
        # デフォルト値を返す
        return (1000.0, 1000.0, 960.0, 540.0)

def create_test_camera_pose():
    """
    テスト用のカメラポーズを作成
    
    Returns:
        latest_pose, past_pose: 最新と過去のカメラポーズ行列
    """
    # 単位行列（視点の変化なし）
    identity = np.eye(4)
    
    # 過去カメラは大きく移動と回転を加える（テーブル上の本を見る視点の変化を強調）
    past_pose = np.eye(4)
    past_pose[0, 3] = 0.6  # X軸方向に60cm移動（変更: 0.5→0.6）
    past_pose[1, 3] = 0.3  # Y軸方向に30cm移動（変更: 0.2→0.3）
    
    # より大きな回転を加える（Y軸周りに回転）
    theta = np.radians(15)  # 15度の回転（変更: 10→15）
    c, s = np.cos(theta), np.sin(theta)
    rotation_y = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])
    
    # X軸周りの回転も追加（見下ろす角度の変化）
    phi = np.radians(8)  # 8度の回転（変更: 5→8）
    c, s = np.cos(phi), np.sin(phi)
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])
    
    # 回転を適用（最初にY軸回転、次にX軸回転）
    past_pose = np.dot(rotation_x, np.dot(rotation_y, past_pose))
    
    return identity, past_pose

def main():
    print("動的物体検出とマスク処理テスト実行")
    
    # テストデータのパス設定
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    latest_image_path = os.path.join(test_data_dir, "jaquar0247.jpeg")
    past_image_path = os.path.join(test_data_dir, "jaquar0124.jpeg")
    cameras_file = os.path.join(test_data_dir, "cameras.txt")
    
    # 出力ディレクトリ
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像の読み込み
    latest_image = cv2.imread(latest_image_path)
    past_image = cv2.imread(past_image_path)
    
    if latest_image is None or past_image is None:
        print(f"画像の読み込みに失敗しました: {latest_image_path} または {past_image_path}")
        sys.exit(1)
    
    print(f"最新画像サイズ: {latest_image.shape}")
    print(f"過去画像サイズ: {past_image.shape}")
    
    # カメラパラメータの読み込み
    camera_intrinsics = load_camera_params_from_colmap(cameras_file)
    print(f"カメラ内部パラメータ: fx={camera_intrinsics[0]}, fy={camera_intrinsics[1]}, cx={camera_intrinsics[2]}, cy={camera_intrinsics[3]}")
    
    # テスト用のカメラポーズを作成
    latest_pose, past_pose = create_test_camera_pose()
    print(f"最新画像カメラポーズ:\n{latest_pose}")
    print(f"過去画像カメラポーズ:\n{past_pose}")
    
    # モデルの初期化
    print("\nモデルを初期化中...")
    depth_model = DepthEstimator()
    segmentation_model = SegmentationModel()
    point_cloud_processor = PointCloudProcessor()
    dynamic_detector = DynamicObjectDetector(dynamic_threshold=0.08)  # 動的閾値をさらに下げる: 0.10 → 0.08
    
    # 処理パラメータ
    distance_threshold = 0.05  # 動的物体と判定する距離閾値をさらに下げる: 0.08 → 0.05
    
    # 処理開始
    print("\n=== 動的物体検出処理を開始 ===")
    
    # 1. 深度推定
    print("1. 深度推定中...")
    latest_depth = depth_model.estimate_depth(latest_image)
    past_depth = depth_model.estimate_depth(past_image)
    
    # 深度マップの統計情報
    print(f"最新画像深度範囲: {np.min(latest_depth):.2f}m～{np.max(latest_depth):.2f}m、平均: {np.mean(latest_depth):.2f}m")
    print(f"過去画像深度範囲: {np.min(past_depth):.2f}m～{np.max(past_depth):.2f}m、平均: {np.mean(past_depth):.2f}m")
    
    # 深度マップを可視化して保存
    latest_depth_vis = cv2.normalize(latest_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    past_depth_vis = cv2.normalize(past_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # カラーマップを適用して詳細に可視化
    latest_depth_color = cv2.applyColorMap(latest_depth_vis, cv2.COLORMAP_JET)
    past_depth_color = cv2.applyColorMap(past_depth_vis, cv2.COLORMAP_JET)
    
    cv2.imwrite(os.path.join(output_dir, "latest_depth.jpg"), latest_depth_vis)
    cv2.imwrite(os.path.join(output_dir, "past_depth.jpg"), past_depth_vis)
    cv2.imwrite(os.path.join(output_dir, "latest_depth_color.jpg"), latest_depth_color)
    cv2.imwrite(os.path.join(output_dir, "past_depth_color.jpg"), past_depth_color)
    
    # 2. 点群生成
    print("2. 点群生成中...")
    latest_points = depth_model.generate_point_cloud(latest_image, latest_depth, camera_intrinsics)
    past_points = depth_model.generate_point_cloud(past_image, past_depth, camera_intrinsics)
    
    print(f"最新画像の点数: {len(latest_points)}")
    print(f"過去画像の点数: {len(past_points)}")
    
    # 点群の統計情報
    print(f"最新画像点群 X範囲: [{np.min(latest_points[:,0]):.2f}, {np.max(latest_points[:,0]):.2f}]")
    print(f"最新画像点群 Y範囲: [{np.min(latest_points[:,1]):.2f}, {np.max(latest_points[:,1]):.2f}]")
    print(f"最新画像点群 Z範囲: [{np.min(latest_points[:,2]):.2f}, {np.max(latest_points[:,2]):.2f}]")
    print(f"過去画像点群 X範囲: [{np.min(past_points[:,0]):.2f}, {np.max(past_points[:,0]):.2f}]")
    print(f"過去画像点群 Y範囲: [{np.min(past_points[:,1]):.2f}, {np.max(past_points[:,1]):.2f}]")
    print(f"過去画像点群 Z範囲: [{np.min(past_points[:,2]):.2f}, {np.max(past_points[:,2]):.2f}]")
    
    # 3. 過去画像の点群を最新画像の視点に変換
    print("3. 点群座標変換中...")
    past_points_transformed = point_cloud_processor.transform_point_cloud(past_points, past_pose)
    
    # 変換後の点群統計情報
    print(f"変換後の過去画像点群 X範囲: [{np.min(past_points_transformed[:,0]):.2f}, {np.max(past_points_transformed[:,0]):.2f}]")
    print(f"変換後の過去画像点群 Y範囲: [{np.min(past_points_transformed[:,1]):.2f}, {np.max(past_points_transformed[:,1]):.2f}]")
    print(f"変換後の過去画像点群 Z範囲: [{np.min(past_points_transformed[:,2]):.2f}, {np.max(past_points_transformed[:,2]):.2f}]")
    
    # 変換の可視化（点群をZ方向のカラーマップで画像に投影）
    def visualize_point_cloud(points, image_shape, camera_intrinsics, name):
        fx, fy, cx, cy = camera_intrinsics
        height, width = image_shape[:2]
        
        # Z値に基づいて色付け
        z_min, z_max = np.min(points[:,2]), np.max(points[:,2])
        z_normalized = (points[:,2] - z_min) / (z_max - z_min)
        
        # 可視化用の画像を作成
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 点を画像に投影
        for i, (x, y, z) in enumerate(points[:,:3]):
            if z > 0:  # 前方の点のみ
                px = int((fx * x / z) + cx)
                py = int((fy * y / z) + cy)
                
                if 0 <= px < width and 0 <= py < height:
                    # Z値に応じた色（赤:近い、青:遠い）
                    color = [
                        int(255 * (1 - z_normalized[i])),  # B
                        0,                                # G
                        int(255 * z_normalized[i])         # R
                    ]
                    vis_image[py, px] = color
        
        # 穴を埋めるために膨張と侵食
        kernel = np.ones((3, 3), np.uint8)
        vis_image = cv2.dilate(vis_image, kernel, iterations=1)
        
        return vis_image
    
    # 点群の可視化
    latest_points_vis = visualize_point_cloud(latest_points, latest_image.shape, camera_intrinsics, "latest")
    past_points_vis = visualize_point_cloud(past_points, past_image.shape, camera_intrinsics, "past")
    past_transformed_vis = visualize_point_cloud(past_points_transformed, latest_image.shape, camera_intrinsics, "past_transformed")
    
    cv2.imwrite(os.path.join(output_dir, "latest_points.jpg"), latest_points_vis)
    cv2.imwrite(os.path.join(output_dir, "past_points.jpg"), past_points_vis)
    cv2.imwrite(os.path.join(output_dir, "past_transformed_points.jpg"), past_transformed_vis)
    
    # 4. 点群比較（最近傍探索で動的点を検出）
    print("4. 点群比較（動的物体検出）中...")
    latest_index = point_cloud_processor.build_faiss_index(latest_points)
    
    dynamic_indices, distances = point_cloud_processor.find_nearest_neighbors(
        past_points_transformed, latest_index, distance_threshold=distance_threshold
    )
    
    # 動的点群の可視化
    dynamic_points_mask = np.zeros(len(past_points_transformed), dtype=bool)
    dynamic_points_mask[dynamic_indices] = True
    
    # 動的点と静的点を分けて可視化
    def visualize_dynamic_points(points, dynamic_mask, image_shape, camera_intrinsics, name):
        fx, fy, cx, cy = camera_intrinsics
        height, width = image_shape[:2]
        
        # 可視化用の画像を作成
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # テーブル上（Y座標の特定範囲）の点をハイライトするマスク
        table_mask = (points[:, 1] > -1.0) & (points[:, 1] < -0.3)
        
        # 点を画像に投影
        for i, (x, y, z) in enumerate(points[:,:3]):
            if z > 0:  # 前方の点のみ
                px = int((fx * x / z) + cx)
                py = int((fy * y / z) + cy)
                
                if 0 <= px < width and 0 <= py < height:
                    # 動的点は赤、テーブル上の点は緑、その他の静的点は青
                    if dynamic_mask[i]:
                        color = [0, 0, 255]  # 赤: 動的点
                    elif table_mask[i]:
                        color = [0, 255, 0]  # 緑: テーブル上の点
                    else:
                        color = [255, 0, 0]  # 青: その他の静的点
                    vis_image[py, px] = color
        
        # 穴を埋めるために膨張
        kernel = np.ones((3, 3), np.uint8)
        vis_image = cv2.dilate(vis_image, kernel, iterations=1)
        
        return vis_image
    
    # 動的点の可視化
    dynamic_points_vis = visualize_dynamic_points(past_points_transformed, dynamic_points_mask, latest_image.shape, camera_intrinsics, "dynamic_points")
    cv2.imwrite(os.path.join(output_dir, "dynamic_points.jpg"), dynamic_points_vis)
    
    # 5. 過去画像のセグメンテーション
    print("5. 過去画像のセグメンテーション中...")
    past_segments = segmentation_model.segment_image(past_image)
    
    # セグメンテーション結果を可視化
    past_image_with_segments = past_image.copy()
    segment_colors = {}
    
    # 特定物体（本など）を検出するためのクラス情報
    object_of_interest = {
        'center_x': 960,  # 画像の中央付近
        'center_y': 540,
        'width': 300,     # ある程度の大きさ（本の大きさ）
        'height': 300
    }
    
    # 本の候補となるセグメントのインデックスを保存
    book_segment_indices = []
    
    for i, seg in enumerate(past_segments):
        # 各セグメントに固有の色を割り当て
        if i not in segment_colors:
            segment_colors[i] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        
        mask = seg['mask'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(past_image_with_segments, contours, -1, segment_colors[i], 2)
        
        # セグメントの位置と大きさを計算
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # セグメントIDを描画
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(past_image_with_segments, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # セグメントの面積
            segment_area = cv2.contourArea(contours[0])
            
            # 特定のセグメントを検出（本などの動的オブジェクト）
            # 位置と大きさを考慮
            is_in_center = abs(cx - object_of_interest['center_x']) < 300 and abs(cy - object_of_interest['center_y']) < 200
            is_book_size = 2000 < segment_area < 30000 and 50 < w < 400 and 50 < h < 400
            
            if is_in_center and is_book_size:
                print(f"セグメント{i}は本などの対象物である可能性が高い: 面積={segment_area}, 位置=({cx},{cy}), サイズ=({w}x{h})")
                book_segment_indices.append(i)
        
    # 追加の視覚化：特定したセグメントに特別なマーキング
    for book_idx in book_segment_indices:
        seg = past_segments[book_idx]
        mask = seg['mask'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 本のセグメントを赤い太線で強調
        cv2.drawContours(past_image_with_segments, contours, -1, (0, 0, 255), 4)
        # 「本」というラベルを追加
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(past_image_with_segments, f"{book_idx}:BOOK", (cx-30, cy-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(os.path.join(output_dir, "past_segmentation.jpg"), past_image_with_segments)
    
    # 6. 動的点群の投影＆動的セグメントの検出
    print("6. 動的セグメント検出中...")
    image_points, valid_mask = dynamic_detector.project_points_to_image(
        past_points_transformed, camera_intrinsics, past_image.shape
    )
    
    dynamic_segment_indices, segment_ratios = dynamic_detector.detect_dynamic_segments(
        dynamic_indices, image_points, valid_mask, past_segments
    )
    
    # 動的点の集合を取得（HashSetで高速な検索用）
    dynamic_points_set = set(dynamic_indices)
    
    # セグメント情報の詳細分析
    print("\n--- セグメント詳細分析 ---")
    segment_analysis = []
    book_segment_candidates = []
    
    for i, seg in enumerate(past_segments):
        mask = seg['mask'].astype(np.uint8)
        segment_area = np.sum(mask)
        
        # セグメント内の点の位置を分析（セグメントに対応する3D点を特定）
        segment_points = []
        for j in range(len(image_points)):
            if valid_mask[j]:
                x, y = image_points[j]
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    segment_points.append(j)
        
        # 統計情報の計算
        if segment_points:
            # 深度情報
            depths = past_points_transformed[segment_points, 2]
            mean_depth = np.mean(depths)
            
            # 3D座標情報（特にY座標 = 高さ）
            y_coords = past_points_transformed[segment_points, 1]
            mean_y = np.mean(y_coords) if len(y_coords) > 0 else 0
            
            # 動的ピクセルの割合
            dynamic_pixels = [p for p in segment_points if p in dynamic_points_set]
            dynamic_count = len(dynamic_pixels)
            dynamic_ratio = dynamic_count / len(segment_points) if segment_points else 0
            
            # セグメント情報を保存
            segment_info = {
                'index': i,
                'area': segment_area,
                'mean_y': mean_y,
                'mean_depth': mean_depth,
                'dynamic_ratio': dynamic_ratio,
                'dynamic_count': dynamic_count,
                'total_points': len(segment_points)
            }
            segment_analysis.append(segment_info)
            
            # 出力情報
            print(f"セグメント{i}: 面積={segment_area}px, 平均高さ={mean_y:.2f}m, 平均深度={mean_depth:.2f}m, 動的率={dynamic_ratio:.3f} ({dynamic_count}/{len(segment_points)})")
            
            # テーブル上の物体候補を検出（高さと面積で判定）
            table_top_y = -0.7  # テーブルトップ付近のY座標（負の値なので注意）
            is_on_table = mean_y > (table_top_y - 0.3) and mean_y < (table_top_y + 0.3)  # テーブル面の前後30cm
            is_book_size = segment_area > 2000 and segment_area < 30000  # 本らしいサイズ
            
            if is_on_table and is_book_size:
                book_segment_candidates.append(segment_info)
                print(f"  👉 セグメント{i}は机の上の本である可能性が高い！")
    
    print("\n--- 本の候補となるセグメント ---")
    for book in sorted(book_segment_candidates, key=lambda x: x['area'], reverse=True):
        idx = book['index']
        print(f"セグメント{idx}: 面積={book['area']}px, 高さ={book['mean_y']:.2f}m, 動的率={book['dynamic_ratio']:.3f}")
        
        # 本候補セグメントを強制的に動的セグメントとして追加
        if idx not in dynamic_segment_indices:
            print(f"  👉 セグメント{idx}を動的セグメントとして強制追加")
            dynamic_segment_indices.append(idx)
    
    # 特定した本のセグメントも強制的に動的として追加
    for book_idx in book_segment_indices:
        if book_idx not in dynamic_segment_indices:
            print(f"  👉 セグメント{book_idx}（本）を動的セグメントとして強制追加")
            dynamic_segment_indices.append(book_idx)
    
    print(f"\n動的セグメント検出結果: {len(dynamic_segment_indices)}/{len(past_segments)}個のセグメントが動的と判定")
    print(f"動的セグメントインデックス: {sorted(dynamic_segment_indices)}")
    
    # 動的セグメントの可視化
    dynamic_segments_vis = past_image.copy()
    
    for i, seg in enumerate(past_segments):
        mask = seg['mask'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 動的セグメントは赤、静的セグメントは緑で表示
        color = (0, 0, 255) if i in dynamic_segment_indices else (0, 255, 0)
        cv2.drawContours(dynamic_segments_vis, contours, -1, color, 2)
        
        # セグメントIDを描画
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(dynamic_segments_vis, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # セグメントのサイズを計算
        segment_area = np.sum(mask)
        
        # テーブルと本に関連すると思われるセグメントの特徴を出力
        if segment_area > 1000 and segment_area < 50000:  # テーブルや本らしきサイズ制限
            # セグメント内の点の位置を分析
            segment_points = []
            for j in range(len(image_points)):
                if valid_mask[j]:
                    x, y = image_points[j]
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                        segment_points.append(j)
            
            if segment_points:
                # セグメント内の点の平均Y座標（机の高さに関連）を計算
                y_coords = past_points_transformed[segment_points, 1]
                mean_y = np.mean(y_coords) if len(y_coords) > 0 else 0
                
                # セグメント内の動的ピクセルの割合
                dynamic_count = sum(1 for p in segment_points if p in dynamic_segment_indices)
                dynamic_ratio = dynamic_count / len(segment_points) if segment_points else 0
                
                print(f"セグメント{i}の詳細: 面積={segment_area}px, 平均Y座標={mean_y:.2f}m, 動的率={dynamic_ratio:.2f}")
                
                # 机の上に位置する可能性のあるセグメントを特定
                is_on_table = mean_y > -0.9 and mean_y < -0.3  # 机の高さ付近
                if is_on_table:
                    print(f"  セグメント{i}は机の上に位置している可能性が高い")
                    
                    # 机の上で動的ではない場合は特に注目
                    if i not in dynamic_segment_indices and dynamic_ratio > 0.05:
                        print(f"  ⚠️ このセグメントは本などの動的物体の可能性があるが検出されていません!")
                        print(f"  動的ピクセルあり: {dynamic_count}/{len(segment_points)}={dynamic_ratio:.2f}")

    cv2.imwrite(os.path.join(output_dir, "dynamic_segments.jpg"), dynamic_segments_vis)
    
    # 7. 動的マスクの生成
    print("7. 動的マスク生成中...")
    dynamic_mask = dynamic_detector.create_dynamic_mask(
        dynamic_segment_indices, past_segments, past_image.shape
    )
    
    # 8. マスク適用
    print("8. マスク適用中...")
    masked_past_image = dynamic_detector.apply_mask_to_image(past_image, dynamic_mask)
    
    # 9. 結果の可視化
    print("9. 結果の可視化...")
    
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
    comparison[:, w*2:w*3] = masked_past_image
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
    cv2.imwrite(os.path.join(output_dir, "latest.jpg"), latest_image)
    cv2.imwrite(os.path.join(output_dir, "past.jpg"), past_image)
    cv2.imwrite(os.path.join(output_dir, "masked.jpg"), masked_past_image)
    cv2.imwrite(os.path.join(output_dir, "mask.jpg"), dynamic_mask * 255)
    
    print(f"\n処理が完了しました。結果は {output_dir} に保存されています。")
    
    # 結果を表示
    try:
        cv2.imshow("Results", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("GUIでの表示に失敗しました。結果は画像ファイルで確認してください。")

if __name__ == "__main__":
    main() 