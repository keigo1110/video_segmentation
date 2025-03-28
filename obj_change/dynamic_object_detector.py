import os
import sys
import cv2
import numpy as np
import time

class DynamicObjectDetector:
    """
    動的物体の検出とマスク処理を行うクラス
    """
    def __init__(self, dynamic_threshold=0.5):
        """
        初期化関数
        
        Args:
            dynamic_threshold: セグメント内の動的ピクセル割合の閾値（この割合を超えるとセグメント全体を動的とみなす）
        """
        self.dynamic_threshold = dynamic_threshold
        print(f"動的物体検出器を初期化しました（動的閾値: {dynamic_threshold}）")
        
    def project_points_to_image(self, points_3d, camera_intrinsics, image_shape):
        """
        3D点群を画像平面に投影する
        
        Args:
            points_3d: 3D点群（N×3の配列）
            camera_intrinsics: カメラの内部パラメータ（fx, fy, cx, cy）
            image_shape: 画像のサイズ（height, width）
            
        Returns:
            画像座標のリスト（N×2の配列）、投影可能なマスク
        """
        fx, fy, cx, cy = camera_intrinsics
        height, width = image_shape[:2]
        
        # 3D点をカメラ座標系に変換（既に変換されていることを前提）
        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # 奥行きが正の場合のみ投影可能
        valid_mask = Z > 0
        
        # 画像平面への投影
        x = (fx * X / Z) + cx
        y = (fy * Y / Z) + cy
        
        # 画像内に収まる点のマスクを作成
        in_image_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        
        # 有効な投影点のマスク（奥行き正かつ画像内）
        valid_projection_mask = valid_mask & in_image_mask
        
        # 画像座標を整数に変換
        image_points = np.column_stack([x, y]).astype(np.int32)
        
        return image_points, valid_projection_mask
        
    def detect_dynamic_segments(self, dynamic_points, image_points, valid_projection_mask, segments):
        """
        動的点群を元にセグメントの動的/静的を判定する
        
        Args:
            dynamic_points: 動的と判定された3D点のインデックス
            image_points: 画像平面に投影された点群座標（N×2の配列）
            valid_projection_mask: 有効な投影点のマスク
            segments: セグメンテーション結果（FastSAMの出力）
            
        Returns:
            動的と判定されたセグメントのインデックスリスト、各セグメントの動的度合い
        """
        # 動的点の画像座標を取得（高速検索のためset型で）
        dynamic_points_set = set(dynamic_points)
        
        # 各セグメントの情報収集
        segment_stats = []
        
        for i, segment in enumerate(segments):
            segment_mask = segment['mask']
            
            # セグメントの面積を計算
            segment_area = np.sum(segment_mask)
            
            # 投影点をセグメントマスクと比較
            segment_points = []  # このセグメント内の有効な点のインデックス
            
            for j in range(len(image_points)):
                # 投影が有効な点のみを対象とする
                if valid_projection_mask[j]:
                    x, y = image_points[j]
                    
                    # 座標がセグメント内にあるか確認
                    if 0 <= y < segment_mask.shape[0] and 0 <= x < segment_mask.shape[1] and segment_mask[y, x]:
                        segment_points.append(j)
            
            # このセグメント内の動的点をカウント
            dynamic_count = sum(1 for p in segment_points if p in dynamic_points_set)
            total_points = len(segment_points)
            
            # セグメント情報を保存
            segment_info = {
                'index': i,
                'area': segment_area,
                'total_points': total_points,
                'dynamic_count': dynamic_count
            }
            
            # 動的度合いの計算（セグメントサイズに応じた調整）
            if total_points > 0:
                # 基本的な動的比率
                segment_info['dynamic_ratio'] = dynamic_count / total_points
                
                # セグメントの大きさによる補正（小さいセグメントは判定を緩く）
                area_factor = min(1.0, 10000 / max(segment_area, 500))
                segment_info['adjusted_ratio'] = segment_info['dynamic_ratio'] * (1.0 + area_factor)
                
                # 絶対的な動的点数の考慮（一定数以上の動的点があれば重視）
                absolute_factor = min(1.0, dynamic_count / 5)  # 5点以上で最大
                segment_info['final_score'] = max(
                    segment_info['adjusted_ratio'],
                    segment_info['dynamic_ratio'] + absolute_factor
                )
            else:
                segment_info['dynamic_ratio'] = 0.0
                segment_info['adjusted_ratio'] = 0.0
                segment_info['final_score'] = 0.0
            
            segment_stats.append(segment_info)
        
        # スコアが高い順にソート（デバッグ用）
        sorted_segments = sorted(segment_stats, key=lambda x: x['final_score'], reverse=True)
        
        # 上位のセグメント情報をログ出力
        print(f"上位の動的セグメント候補:")
        for i, segment in enumerate(sorted_segments[:10]):  # 上位10件
            idx = segment['index']
            print(f"  セグメント {idx}: スコア={segment['final_score']:.3f}, 動的比率={segment['dynamic_ratio']:.3f} ({segment['dynamic_count']}/{segment['total_points']}点), 面積={segment['area']}px")
        
        # 動的と判定されたセグメントのインデックスを取得
        # 改良した判定ロジック: final_scoreが閾値を超えるか、動的点の割合と絶対数を考慮
        dynamic_segment_indices = []
        
        for segment in segment_stats:
            idx = segment['index']
            
            # 動的と判定する条件
            is_dynamic = False
            
            # 条件1: 調整後のスコアが閾値を超える
            if segment['final_score'] > self.dynamic_threshold:
                is_dynamic = True
                
            # 条件2: 小さいセグメントでも、一定以上の動的点が含まれる
            elif segment['dynamic_count'] >= 3 and segment['dynamic_ratio'] > self.dynamic_threshold * 0.5:
                is_dynamic = True
                
            # 条件3: サイズが大きくても動的点の割合が一定以上
            elif segment['area'] > 10000 and segment['dynamic_ratio'] > self.dynamic_threshold * 0.8:
                is_dynamic = True
                
            # 条件を満たしたら追加
            if is_dynamic:
                dynamic_segment_indices.append(idx)
        
        # 各セグメントの動的度合いを返す（診断用）
        segment_ratios = {info['index']: info['final_score'] for info in segment_stats}
        
        return dynamic_segment_indices, segment_ratios
        
    def create_dynamic_mask(self, dynamic_segment_indices, segments, image_shape):
        """
        動的セグメントからマスク画像を作成する
        
        Args:
            dynamic_segment_indices: 動的と判定されたセグメントのインデックス
            segments: セグメンテーション結果
            image_shape: 元画像のサイズ
            
        Returns:
            動的物体のマスク画像（2値画像）
        """
        # 空のマスクを作成
        dynamic_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 動的セグメントのマスクを統合
        for idx in dynamic_segment_indices:
            if idx < len(segments):
                seg_mask = segments[idx]['mask'].astype(np.uint8)
                
                # マスクのサイズが画像と一致しない場合はリサイズ
                if seg_mask.shape != image_shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (image_shape[1], image_shape[0]))
                    
                # マスクを論理和で統合
                dynamic_mask = cv2.bitwise_or(dynamic_mask, seg_mask)
        
        return dynamic_mask
        
    def apply_mask_to_image(self, image, mask, mask_color=(0, 0, 0)):
        """
        画像にマスクを適用する
        
        Args:
            image: 元画像
            mask: 適用するマスク（2値画像）
            mask_color: マスク部分の色（デフォルトは黒）
            
        Returns:
            マスク適用後の画像
        """
        # 画像のコピーを作成
        masked_image = image.copy()
        
        # マスク部分を指定色で塗りつぶし
        if len(image.shape) == 3:  # カラー画像の場合
            masked_image[mask > 0] = mask_color
        else:  # グレースケール画像の場合
            masked_image[mask > 0] = mask_color[0]
            
        return masked_image 