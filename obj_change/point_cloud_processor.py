import numpy as np
import torch
import faiss
import time

class PointCloudProcessor:
    """
    点群の処理（座標変換、比較など）を行うクラス
    """
    def __init__(self, use_gpu=True):
        """
        初期化関数
        
        Args:
            use_gpu: GPUを使用するかどうか（Faissでの高速処理に使用）
        """
        try:
            self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
            
            if self.use_gpu:
                print(f"Faiss GPU検出: {faiss.get_num_gpus()}個のGPUが利用可能です")
            else:
                print("Faiss GPUが検出されないため、CPU版を使用します")
        except:
            self.use_gpu = False
            print("Faiss GPUサポートがないため、CPU版を使用します")
            
    def transform_point_cloud(self, points, transform_matrix):
        """
        点群を与えられた変換行列で変換する
        
        Args:
            points: 点群データ（N×3以上の配列、XYZ座標が最初の3列）
            transform_matrix: 4×4変換行列（回転と平行移動を含む）
            
        Returns:
            変換された点群
        """
        # 点の座標部分のみを取得
        xyz = points[:, :3].copy()
        
        # 同次座標に変換（N×4の行列、最後の要素は1）
        homogeneous_points = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        
        # 変換行列を適用
        transformed_points = np.dot(homogeneous_points, transform_matrix.T)
        
        # 同次座標から3次元座標に戻す
        transformed_xyz = transformed_points[:, :3]
        
        # 結果の点群を作成（元の色情報などは保持）
        result = points.copy()
        result[:, :3] = transformed_xyz
        
        return result
        
    def build_faiss_index(self, points, dimension=3):
        """
        Faissインデックスを構築する
        
        Args:
            points: インデックスを構築する点群（N×D行列）
            dimension: 点の次元数（デフォルトは3D座標のXYZ）
            
        Returns:
            構築されたFaissインデックス
        """
        # 3D座標部分のみを取得
        points_data = points[:, :dimension].copy().astype(np.float32)
        
        # Faissインデックスの作成
        index = faiss.IndexFlatL2(dimension)
        
        # GPUを使用する場合
        if self.use_gpu:
            try:
                # GPU版インデックスに変換
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                print(f"GPU版Faissの使用に失敗しました: {e}")
                print("CPU版Faissを使用します")
        
        # 点群をインデックスに追加
        index.add(points_data)
        
        return index
        
    def find_nearest_neighbors(self, query_points, reference_index, k=1, distance_threshold=0.2):
        """
        クエリ点に対して参照点群から最近傍点を検索し、動的物体を検出する
        
        Args:
            query_points: クエリ点群（N×D行列）
            reference_index: 参照点群のFaissインデックス
            k: 検索する最近傍点の数
            distance_threshold: 動的物体と判定する距離の閾値（メートル）
            
        Returns:
            動的点のインデックス、最近傍点との距離
        """
        # クエリ点の3D座標部分のみを取得
        query_data = query_points[:, :3].copy().astype(np.float32)
        
        # 最近傍検索の実行
        start_time = time.time()
        distances, indices = reference_index.search(query_data, k)
        search_time = time.time() - start_time
        
        print(f"最近傍検索完了: {query_data.shape[0]}点に対して{search_time:.4f}秒")
        
        # 距離閾値を超える点を動的物体として検出
        dynamic_mask = distances[:, 0] > distance_threshold**2  # L2距離の二乗値で比較
        
        # 動的点のインデックスを取得
        dynamic_indices = np.where(dynamic_mask)[0]
        
        print(f"動的物体点の検出: 全{query_data.shape[0]}点中{len(dynamic_indices)}点が閾値（{distance_threshold}m）を超過")
        
        return dynamic_indices, distances 