import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import time

# FastSAMのディレクトリをシステムパスに追加
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "FastSAM"))

# FastSAMのインポート
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics パッケージがインストールされていません。以下のコマンドでインストールしてください:")
    print("pip install ultralytics")
    sys.exit(1)

class SegmentationModel:
    """
    FastSAMを使用した高速セグメンテーションを行うクラス
    """
    def __init__(self, model_path=None, conf_threshold=0.4, iou_threshold=0.9):
        """
        初期化関数
        
        Args:
            model_path: FastSAMモデルのパス（Noneの場合は自動的に探索）
            conf_threshold: 信頼度の閾値
            iou_threshold: IoU（重なり）の閾値
        """
        print("セグメンテーションモデルを初期化中...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 絶対パスからの現在のディレクトリ
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # モデルパスの自動探索
        if model_path is None:
            # モデルファイルを探索する順序：
            search_paths = [
                os.path.join(current_dir, "..", "FastSAM-x.pt"),  # リポジトリルート
                os.path.join(current_dir, "..", "FastSAM", "FastSAM-x.pt"),  # FastSAMディレクトリ
                "/home/rkmtlabkei/video_segmentation/FastSAM-x.pt",  # 絶対パス
                "/home/rkmtlabkei/video_segmentation/FastSAM/FastSAM-x.pt",  # 絶対パス
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"モデルファイルを発見: {os.path.abspath(model_path)}")
                    break
            else:
                print("FastSAM-x.ptが見つかりませんでした。リポジトリのルートにモデルファイルがあるか確認してください。")
                sys.exit(1)
        
        try:
            print(f"モデルを読み込み中: {model_path}")
            self.model = YOLO(model_path)
            print(f"FastSAMモデルを初期化しました: {model_path}")
            print(f"デバイス: {self.device}")
        except Exception as e:
            print(f"モデルの初期化に失敗しました: {e}")
            sys.exit(1)
            
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def segment_image(self, image, return_visualization=False):
        """
        画像のセグメンテーションを実行
        
        Args:
            image: NumPy配列またはファイルパスの画像
            return_visualization: 視覚化結果も返すかどうか
            
        Returns:
            segments: セグメントの情報（マスク、信頼度など）
            vis_image: (オプション) 視覚化された画像
        """
        start_time = time.time()

        # 画像の読み込み
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"画像ファイルが見つかりません: {image}")
            img_path = image
        elif isinstance(image, np.ndarray):
            # 一時的に画像を保存
            temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_image_for_segmentation.jpg")
            cv2.imwrite(temp_path, image)
            img_path = temp_path
        else:
            raise TypeError("画像はファイルパスまたはNumPy配列である必要があります")
            
        # セグメンテーションの実行
        try:
            results = self.model(
                source=img_path,
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                retina_masks=True,
                verbose=False
            )
        except Exception as e:
            print(f"セグメンテーション実行中にエラーが発生しました: {e}")
            # エラーが発生した場合は空の結果を返す
            process_time = time.time() - start_time
            print(f"セグメンテーション失敗: {process_time:.4f}秒")
            return []
        
        # 一時ファイルの削除
        if isinstance(image, np.ndarray) and os.path.exists(temp_path):
            os.remove(temp_path)
            
        process_time = time.time() - start_time
        print(f"セグメンテーション完了: {process_time:.4f}秒")
        
        # セグメント情報を抽出
        segments = []
        
        for result in results:
            # マスク情報が存在する場合
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                for i, mask in enumerate(result.masks.data):
                    mask_np = mask.cpu().numpy()
                    
                    if i < len(result.boxes):
                        bbox = result.boxes.data[i].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
                        confidence = bbox[4]
                    else:
                        confidence = 0.5  # デフォルト値
                        bbox = np.array([0, 0, mask_np.shape[1], mask_np.shape[0], confidence, 0])
                    
                    segments.append({
                        'mask': mask_np,
                        'confidence': confidence,
                        'bbox': bbox[:4]
                    })
        
        print(f"検出したセグメント数: {len(segments)}")
        
        # 視覚化
        if return_visualization:
            if isinstance(image, str):
                vis_image = cv2.imread(image)
            else:
                vis_image = image.copy()
                
            # セグメントの可視化
            for seg in segments:
                mask = seg['mask'].astype(np.uint8)
                bbox = seg['bbox'].astype(int)
                
                # マスクの輪郭を抽出
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 輪郭を描画
                cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
                
                # バウンディングボックスを描画
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                
                # 信頼度を表示
                cv2.putText(vis_image, f"{seg['confidence']:.2f}", 
                           (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            return segments, vis_image
        
        return segments
        
    def create_mask_from_segments(self, segments, image_shape):
        """
        セグメント情報から2値マスクを作成
        
        Args:
            segments: セグメント情報のリスト
            image_shape: 元画像のサイズ (height, width)
            
        Returns:
            マスク画像（2値画像）
        """
        # 空のマスクを作成
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 各セグメントのマスクを統合
        for seg in segments:
            seg_mask = seg['mask'].astype(np.uint8)
            
            # マスクのサイズを元画像に合わせる
            if seg_mask.shape != image_shape[:2]:
                seg_mask = cv2.resize(seg_mask, (image_shape[1], image_shape[0]))
                
            # マスクを結合（論理和）
            mask = cv2.bitwise_or(mask, seg_mask)
            
        return mask 