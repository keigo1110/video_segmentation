import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimator:
    """
    深度推定を行うクラス
    """
    def __init__(self, model_type="depth-anything/Depth-Anything-V2-Base"):
        """
        初期化関数
        
        Args:
            model_type: 深度推定モデルのタイプ
                - "depth-anything/Depth-Anything-V2-Small" - Depth Anything V2（小）
                - "depth-anything/Depth-Anything-V2-Base" - Depth Anything V2（中）
                - "depth-anything/Depth-Anything-V2-Large" - Depth Anything V2（大）
                - "Intel/dpt-large" - Intel DPTモデル（大）
        """
        print("深度推定モデルを初期化中...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_type)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_type).to(self.device)
            print(f"深度推定モデルを初期化しました: {model_type}")
            print(f"デバイス: {self.device}")
        except Exception as e:
            print(f"モデルの初期化に失敗しました: {e}")
            print("代替モデルを試みます...")
            
            # 代替モデルを試す
            fallback_models = [
                "Intel/dpt-large",
                "Intel/dpt-swinv2-tiny-256",
                "LiheYoung/depth-anything-base-hf"
            ]
            
            for model in fallback_models:
                try:
                    print(f"モデル {model} を試しています...")
                    self.processor = AutoImageProcessor.from_pretrained(model)
                    self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device)
                    print(f"深度推定モデルを初期化しました: {model}")
                    print(f"デバイス: {self.device}")
                    break
                except Exception as e2:
                    print(f"モデル {model} の初期化に失敗しました: {e2}")
            else:
                print("利用可能な深度推定モデルが見つかりませんでした。")
                sys.exit(1)
            
    def estimate_depth(self, image):
        """
        画像から深度マップを推定する
        
        Args:
            image: NumPy配列またはファイルパスの画像
            
        Returns:
            深度マップ（NumPy配列）
        """
        # 画像の読み込み
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image}")
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError("画像はファイルパスまたはNumPy配列である必要があります")
        
        # 深度推定の実行
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # 深度マップの正規化処理
            depth_map = predicted_depth.squeeze().cpu().numpy()
            
            # スケール調整 (メートル単位に変換)
            # 深度推定モデルの出力は相対的な深度なので、実際のスケールに調整する必要がある
            # 0-10メートルの範囲にスケーリング（相対的な深度から実際の距離への簡易変換）
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_range = depth_max - depth_min
            
            if depth_range > 0:
                # 一般的な室内シーンの深度範囲を仮定（0-10m）
                depth_map = 10.0 * (depth_map - depth_min) / depth_range
            else:
                # 範囲がない場合（平坦な深度マップ）
                depth_map = np.ones_like(depth_map) * 5.0  # 中間値を設定
            
            return depth_map
            
    def generate_point_cloud(self, image, depth_map, camera_intrinsics):
        """
        深度マップから3D点群を生成する
        
        Args:
            image: RGB画像（NumPy配列）
            depth_map: 深度マップ（NumPy配列）
            camera_intrinsics: カメラの内部パラメータ（fx, fy, cx, cy）
            
        Returns:
            点群（N×6の配列: X, Y, Z, R, G, B）
        """
        # カメラパラメータの展開
        fx, fy, cx, cy = camera_intrinsics
        
        # 画像サイズ取得
        height, width = depth_map.shape
        
        # 座標グリッドの生成
        v, u = np.mgrid[0:height, 0:width]
        
        # 画像座標から3D座標への変換
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 有効な深度値（0より大きい）のマスク
        mask = z > 0
        
        # 点群の作成
        points = np.zeros((np.sum(mask), 6), dtype=np.float32)
        points[:, 0] = x[mask]  # X座標
        points[:, 1] = y[mask]  # Y座標
        points[:, 2] = z[mask]  # Z座標
        
        # カラー情報の追加
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
            
        points[:, 3] = img[..., 2][mask]  # R
        points[:, 4] = img[..., 1][mask]  # G
        points[:, 5] = img[..., 0][mask]  # B
        
        return points 