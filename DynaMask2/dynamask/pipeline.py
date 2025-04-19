import cv2
import numpy as np
import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple

# DynaMask モジュールをインポート (相対インポート)
from .config import DynaMaskConfig
from . import motion
from . import detection
from . import filtering
from . import io
from . import utils # utils をインポート

# --- 外部依存ライブラリ --- #
# PyTorch (オプションだがCUDA利用に推奨)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# FastSAM/video_segment (必須依存)
FASTSAM_AVAILABLE = False
video_segment: Optional[Any] = None
try:
    # ここで FastSAM/video_segment をインポート
    # PYTHONPATH が通っているか、インストールされていることを期待
    # 例: from FastSAM import video_segment (環境に合わせて調整が必要)
    # パッケージ化後は、setup.py での依存関係指定や、
    # video_segment を dynamask パッケージ内に含める等の方法で解決する
    # logging の前に import するのが一般的
    # import video_segment # 仮のインポート名
    from . import video_segment # パッケージ内からの相対インポートに変更
    FASTSAM_AVAILABLE = True
except ImportError as e:
    # loggerが未定義なので print を使用
    print(f"[CRITICAL] 必須依存ライブラリ 'video_segment' が見つかりません: {e}")
    # raise e # 実行前に終了させる場合は例外を送出

# ロガー設定
logger = logging.getLogger(__name__)

# YOLO/MediaPipe 利用可能フラグ (detection モジュールから参照)
from .detection import YOLO_AVAILABLE, MEDIAPIPE_AVAILABLE

# --- ヘルパー関数 --- #
def _wait_for_future(future: Optional[ThreadPoolExecutor]):
    """非同期タスクの結果を安全に取得する"""
    if future is not None:
        try:
            return future.result()
        except Exception as e:
            logger.error(f"非同期処理でエラーが発生しました: {e}", exc_info=True)
            return None
    return None

def _initialize_fastsam_config(config: DynaMaskConfig):
    """DynaMaskConfig に基づいて FastSAM 設定を初期化 (必要な場合)"""
    if config.fastsam_config is None and FASTSAM_AVAILABLE:
        logger.info("config.fastsam_config が未設定のため、DynaMaskConfig から初期化します。")
        try:
            # video_segment.Config の引数を DynaMaskConfig から設定
            # !!! 注意: video_segment.Config の正確な引数リストに合わせて要調整 !!!
            # video_segment.Config が存在するか確認
            if not hasattr(video_segment, 'Config'):
                 logger.error("video_segment モジュールに Config クラスが見つかりません。")
                 return

            fastsam_args = {
                'input_type': config.input_type,
                'input_path': config.input_path,
                'image_pattern': config.image_pattern,
                # output_dir は DynaMask の出力ディレクトリ下に設定
                'output_dir': os.path.join(config.output_dir, "fastsam_output") if config.output_dir else "fastsam_output",
                # 下記は video_segment.Config または load_model で設定される可能性あり
                'model_name': config.fastsam_model_name, # configから取得
                'confidence': config.fastsam_confidence, # configから取得
                'imgsz': config.imgsz, # configから取得
                # 'device': device # これは load_model で設定されるか、別途設定
            }
            # video_segment.Config の引数に合わせて調整 (不明な引数はエラーになる可能性)
            # video_segment.Config が全ての引数を受け入れるか確認が必要
            config.fastsam_config = video_segment.Config(**fastsam_args)
            logger.info(f"生成された FastSAM 設定: {config.fastsam_config}")
        except TypeError as e:
             logger.error(f"FastSAM 設定の初期化中に引数エラー: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"FastSAM 設定の初期化中に予期せぬエラー: {e}", exc_info=True)

# --- メインパイプライン関数 --- #
def run_dynamask(config: DynaMaskConfig) -> Optional[str]:
    """DynaMask パイプラインを実行するメイン関数

    Args:
        config: DynaMaskConfig オブジェクト

    Returns:
        Optional[str]: 成功した場合は出力ディレクトリのパス、失敗した場合は None
    """

    if not FASTSAM_AVAILABLE:
         logger.critical("必須依存ライブラリ 'video_segment' が利用できません。処理を中止します。")
         return None

    start_pipeline_time = time.time()
    logger.info("DynaMask パイプラインを開始します...")
    logger.info(f"使用する設定: {config}") # Configの内容を表示

    # 1. 出力ディレクトリの準備
    if not io.setup_output_directories(config):
        logger.critical("出力ディレクトリの準備に失敗しました。処理を中止します。")
        return None

    # 2. デバイス設定 (CUDA)
    use_cuda = False
    device = 'cpu'
    if TORCH_AVAILABLE:
        try:
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                # TODO: マルチGPU対応やデバイス指定を config から行うように改善可能
                device = 'cuda:0' # シングルGPUを想定
                logger.info(f"CUDA GPUを検出: {torch.cuda.get_device_name(0)}")
                # torch.backends.cudnn.benchmark = True # 入力サイズが変動する場合はFalse推奨
            else:
                 logger.info("CUDA GPU が見つかりません。CPUを使用します。")
        except Exception as e:
             logger.error(f"PyTorch/CUDA の初期化中にエラー: {e}", exc_info=True)
             logger.warning("CPUで実行を試みます。")
             use_cuda = False
             device = 'cpu'
    else:
        logger.info("PyTorch が見つからないため、CPUを使用します。")

    # 3. モデルのロード
    # 3.1 FastSAMモデル
    _initialize_fastsam_config(config) # 必要ならFastSAM設定を初期化
    if config.fastsam_config is None:
        logger.critical("FastSAM設定が利用できません。処理を中止します。")
        return None

    # FastSAM Configにデバイス情報を設定（存在すれば）
    if hasattr(config.fastsam_config, 'device'):
        config.fastsam_config.device = device
        logger.info(f"FastSAM Config のデバイスを '{device}' に設定しました。")
    # else:
        # logger.warning("FastSAM Config に 'device' 属性がありません。デバイス設定をスキップします。")
        # -> video_segment.load_model 側で処理される可能性もあるため、Warningは抑制

    fastsam_model = None
    try:
        logger.info("FastSAMモデルをロード中...")
        if not hasattr(video_segment, 'load_model'):
            logger.critical("video_segment モジュールに load_model 関数が見つかりません。")
            return None
        fastsam_model = video_segment.load_model(config.fastsam_config)
        if fastsam_model is None:
            logger.critical("FastSAMモデルのロードに失敗しました (Noneが返されました)。")
            return None
        logger.info("FastSAMモデルのロード完了。")
    except Exception as e:
        logger.critical(f"FastSAMモデルのロード中に致命的なエラーが発生しました: {e}", exc_info=True)
        return None

    # 3.2 YOLOモデル (人間検出用)
    yolo_model = None
    if config.use_yolo:
        if YOLO_AVAILABLE:
            try:
                # YOLOのインポートは detection モジュール内で行われるべきだが、
                # ここでモデルをインスタンス化するため、ここでもインポートが必要。
                # detection モジュール側と二重管理にならないよう注意。
                from ultralytics import YOLO # 直接インポート
                logger.info(f"YOLOv8モデル ({config.yolo_model}) をロード中... (デバイス: {device})")
                # モデルロード時にデバイスを指定
                yolo_model = YOLO(config.yolo_model).to(device)
                logger.info("YOLOv8モデルのロード完了。")
            except ImportError:
                 # このエラーは通常、上の try...except で捕捉されるはずだが念のため
                 logger.error("ultralytics ライブラリのインポートに失敗しました。")
                 config.use_yolo = False # YOLO利用を無効化
            except Exception as e:
                logger.error(f"YOLOv8モデル ({config.yolo_model}) のロードエラー: {e}", exc_info=True)
                logger.warning("YOLOのロードに失敗したため、YOLOによる人間検出は無効になります。")
                config.use_yolo = False # エラー時もYOLO利用を無効化
        else:
            logger.warning("YOLO が設定で有効になっていますが、ライブラリが見つかりません。YOLO検出は無効になります。")
            config.use_yolo = False # ライブラリがない場合は無効化

    # 3.3 COLMAPデータのロード
    colmap_cameras: Dict[int, Dict] = {}
    colmap_images_by_id: Dict[int, Dict] = {}
    colmap_images_by_name: Dict[str, int] = {}
    if config.colmap_cameras_path and config.colmap_images_path:
        logger.info(f"COLMAPデータをロード中: カメラ={config.colmap_cameras_path}, 画像={config.colmap_images_path}")
        colmap_cameras = io.load_colmap_cameras(config.colmap_cameras_path)
        colmap_images_by_id, colmap_images_by_name = io.load_colmap_images(config.colmap_images_path)
        if not colmap_cameras or not colmap_images_by_id:
            logger.warning("COLMAPデータのロードに一部失敗しました。COLMAP自己運動分離は無効になる可能性があります。")
        else:
            logger.info(f"{len(colmap_cameras)} 個のカメラと {len(colmap_images_by_id)} 個の画像ポーズをロードしました。")
    else:
        logger.info("COLMAPファイルのパスが設定されていないため、COLMAPデータはロードされません。")

    # 4. 入力ソースの初期化 (video_segment を利用)
    source_info: Optional[Dict[str, Any]] = None
    try:
        logger.info("入力ソースを初期化中...")
        if not hasattr(video_segment, 'initialize_input_source'):
             logger.critical("video_segment モジュールに initialize_input_source 関数が見つかりません。")
             return None
        source_info = video_segment.initialize_input_source(config.fastsam_config)
        if source_info is None or not all(k in source_info for k in ['width', 'height', 'fps']):
             logger.critical(f"入力ソースの初期化に失敗しました、または必要な情報 (width, height, fps) が不足しています。Source info: {source_info}")
             return None
        width, height = source_info['width'], source_info['height']
        fps = source_info.get('fps', 30.0) # FPSが取得できない場合のデフォルト値
        logger.info(f"入力ソース初期化完了: {width}x{height} @ {fps:.2f} fps")
    except Exception as e:
        logger.critical(f"入力ソースの初期化中に致命的なエラー: {e}", exc_info=True)
        return None

    # 5. 出力動画ライターの初期化
    video_writer = io.initialize_video_writer(config, width, height, fps)
    if video_writer is None:
        logger.warning("出力動画ライターの初期化に失敗しました。動画ファイルは保存されません。")
        # 動画保存は必須ではないかもしれないので、処理は続行する

    # --- ここまでが初期化処理 ---
    logger.info("初期化完了。フレーム処理を開始します。")

    # 6. フレーム処理ループの準備
    executor = ThreadPoolExecutor(max_workers=6) # ワーカー数を設定 (Configurable?)
    prev_frame: Optional[np.ndarray] = None
    prev_frame_info: Optional[Dict] = None # 前フレームのCOLMAP情報
    prev_hand_positions: List[Tuple[int, int]] = [] # 前フレームの手の位置
    prev_result: Optional[Any] = None
    dynamic_history: Dict[int, float] = {} # セグメントID -> 動的カウント
    human_segments: set[int] = set()
    motion_history: List[np.ndarray] = []
    human_detection_history: Dict[int, float] = {} # セグメントID -> 人間判定カウント (floatに)
    human_confidence_history = np.zeros((height, width), dtype=np.float32)

    # 処理時間計測用
    frame_idx = 0
    processing_times: List[float] = []

    # 非同期処理用フューチャー
    future_human_detection: Optional[ThreadPoolExecutor] = None
    future_segment_result: Optional[ThreadPoolExecutor] = None
    future_motion_mask: Optional[ThreadPoolExecutor] = None

    # バッファ (必要なら)
    # cv_buffer = {
    #     'motion_mask': np.zeros((height, width), dtype=np.uint8),
    #     'human_mask': np.zeros((height, width), dtype=np.uint8),
    #     'temp_mask': np.zeros((height, width), dtype=np.uint8)
    # }

    try:
        # --- フレーム処理ループ開始 ---
        while True:
            loop_start_time = time.time()

            # フレームを取得 (video_segment を利用)
            ret, frame = False, None
            try:
                 if not hasattr(video_segment, 'get_next_frame'):
                     logger.error("video_segment に get_next_frame 関数がありません。ループを終了します。")
                     break
                 ret, frame = video_segment.get_next_frame(source_info, frame_idx)
            except Exception as e:
                 logger.error(f"フレーム {frame_idx} の取得中にエラー: {e}", exc_info=True)
                 break # エラー時はループ終了

            if not ret or frame is None:
                logger.info("入力ソースの終端に到達しました。")
                break # ループ終了

            # フレームサイズのチェック (動的に変わる可能性？)
            if frame.shape[0] != height or frame.shape[1] != width:
                logger.warning(f"フレーム {frame_idx} のサイズが初期サイズ ({width}x{height}) と異なります: {frame.shape[1]}x{frame.shape[0]}。スキップします。")
                frame_idx += 1
                continue # サイズが異なるフレームは処理できない可能性

            current_frame_time_start = time.time()

            # --- 最初のフレーム処理 ---
            if prev_frame is None:
                logger.info("最初のフレームを処理中...")
                prev_frame = frame.copy()

                # 最初のフレームのセグメント化 (同期処理)
                try:
                     if not hasattr(video_segment, 'predict_with_model'):
                         logger.error("video_segment に predict_with_model 関数がありません。処理を中断します。")
                         break
                     prev_result = video_segment.predict_with_model(
                         fastsam_model, prev_frame, config.fastsam_config
                     )
                except Exception as e:
                     logger.error(f"最初のフレームのセグメント化でエラー: {e}", exc_info=True)
                     # prev_result は None のまま

                # 最初のフレームのマスク保存 (すべて静的)
                if config.save_masks:
                    mask_image = np.ones((height, width), dtype=np.uint8) * 255
                    io.save_mask_image(mask_image, frame_idx, config, source_info)

                # 次のフレームの準備 (非同期)
                next_frame = None
                try:
                    if hasattr(video_segment, 'peek_next_frame'):
                        ret_next, next_frame_peek = video_segment.peek_next_frame(source_info)
                        if ret_next and next_frame_peek is not None:
                            next_frame = next_frame_peek.copy() # ピーキングなのでコピーが必要
                    else:
                         logger.warning("video_segment に peek_next_frame 関数がありません。先行処理は無効です。")

                    if next_frame is not None:
                         # 人間検出 (次フレーム)
                         if config.use_pose_detection or config.use_yolo:
                             future_human_detection = executor.submit(
                                 detection.detect_humans_and_hands, next_frame, config, yolo_model
                             )
                         # セグメント化 (次フレーム)
                         if hasattr(video_segment, 'predict_with_model'):
                             future_segment_result = executor.submit(
                                 video_segment.predict_with_model,
                                 fastsam_model, next_frame, config.fastsam_config
                             )
                except Exception as e:
                     logger.error(f"次フレームの先行処理の準備中にエラー: {e}", exc_info=True)

                frame_idx += 1
                proc_time = time.time() - current_frame_time_start
                logger.info(f"初期フレーム処理完了。時間: {proc_time:.4f}秒")
                continue # 次のループへ

            # --- 2フレーム目以降の処理 ---
            current_frame = frame.copy()

            # --- フレーム処理のメイン部分 ---
            frame_start_time = time.time()

            # 6.1 現在フレーム情報の取得 (COLMAP用)
            current_frame_info: Optional[Dict] = None
            current_camera_params: Optional[Dict] = None
            current_filename: Optional[str] = None
            if colmap_images_by_name and source_info:
                 # source_info から現在のファイル名を取得する方法を確認・実装が必要
                 # 例: _generate_output_filename が使えるかもしれない
                 # current_filename = io._generate_output_filename(frame_idx, config.input_type, source_info).replace('.png', '') # 仮
                 # 正確なファイル名取得ロジックが必要
                 if config.input_type == "images" and "file_list" in source_info and frame_idx < len(source_info["file_list"]):
                     current_filepath = source_info["file_list"][frame_idx]
                     current_filename = os.path.basename(current_filepath)
                 elif config.input_type == "video":
                     # 動画の場合、フレーム番号からファイル名を推定するのは難しい
                     # COLMAPを使う場合は連番画像入力が推奨される
                     pass # current_filename は None のまま

                 if current_filename:
                     image_id = colmap_images_by_name.get(current_filename)
                     if image_id:
                         current_frame_info = colmap_images_by_id.get(image_id)
                         if current_frame_info:
                             camera_id = current_frame_info.get('camera_id')
                             if camera_id:
                                 current_camera_params = colmap_cameras.get(camera_id)
                     # if current_frame_info is None:
                         # logger.debug(f"フレーム {frame_idx} ({current_filename}) のCOLMAP情報が見つかりません。")


            # 6.2 動き検出 (非同期実行)
            future_motion_mask = None
            if prev_frame is not None:
                future_motion_mask = executor.submit(
                    motion.detect_motion_optical_flow,
                    config,
                    current_frame.copy(), # コピーを渡す
                    prev_frame.copy(),    # コピーを渡す
                    current_frame_info,   # 現在フレームのCOLMAP情報
                    prev_frame_info,      # 前フレームのCOLMAP情報
                    current_camera_params,# 対応するカメラパラメータ
                    prev_hand_positions   # 前フレームの手の位置
                )

            # 6.3 人間検出 (非同期実行)
            # 既存の人間検出 (YOLO/MediaPipe) の呼び出しを非同期に
            human_detection_results = None
            if config.use_pose_detection or config.use_yolo:
                human_detection_results = _wait_for_future(future_human_detection)
                if human_detection_results is None:
                    logger.warning("人間検出結果の取得に失敗しました。空の結果を使用します。")
                    human_detection_results = {'human_mask': np.zeros((height, width), dtype=np.uint8), 'hand_positions': []}

            # --- [修正] segmentation_mask を事前に初期化 --- #
            segmentation_mask = None # ここで初期化
            num_segments = 0       # num_segments も初期化
            # --- 修正ここまで --- #

            # 6.4 FastSAMによるセグメンテーション (同期実行 or 前回の結果を利用)
            segment_result = None
            if prev_result is None:
                # 最初のフレームは同期実行
                if hasattr(video_segment, 'predict_with_model'):
                    logger.debug("最初のフレームのセグメンテーションを同期実行します。")
                    segment_result = video_segment.predict_with_model(fastsam_model, current_frame, config.fastsam_config)
                else:
                    logger.error("video_segment に predict_with_model 関数がありません。")
                    break # 致命的エラー
            else:
                # 前回の非同期結果を取得
                segment_result = _wait_for_future(future_segment_result)
                if segment_result is None:
                     logger.warning("前回のセグメンテーション結果の取得に失敗しました。")
                     # break または処理を続行するかどうか？ (ここでは続行)

            # 6.5 次のフレームのセグメンテーションを非同期で開始 (結果が得られた後)
            if segment_result is not None:
                if hasattr(video_segment, 'predict_with_model'):
                     future_segment_result = executor.submit(
                         video_segment.predict_with_model,
                         fastsam_model,
                         current_frame.copy(), # 次のループで使う現在のフレーム
                         config.fastsam_config
                     )
                else:
                     logger.error("video_segment に predict_with_model 関数がありません。(非同期開始部分)")
                     # ここで break するか？

            # 6.6 非同期処理の結果を待機
            motion_mask = _wait_for_future(future_motion_mask)
            human_detection_results = _wait_for_future(future_human_detection)

            # 結果が None の場合のデフォルト値
            if motion_mask is None:
                motion_mask = np.zeros((height, width), dtype=np.uint8)
                logger.warning("動きマスクの取得に失敗しました。空のマスクを使用します。")
            if human_detection_results is None:
                 human_detection_results = {'human_mask': np.zeros((height, width), dtype=np.uint8), 'hand_positions': []}
                 logger.warning("人間検出結果がありません (エラーまたは検出なし)。デフォルト値を使用します。")

            # 6.7 結果の解析と動的セグメントの判定
            # --- [修正] current_hand_positions を定義 --- #
            current_hand_positions = human_detection_results.get('hand_positions', []) # 現在の手の位置を取得
            # --- 修正ここまで --- #
            dynamic_indices: List[int] = \
                filtering.filter_dynamic_segments(
                    segment_result,           # 1. segments (予測結果)
                    motion_mask,              # 2. motion_mask
                    (height, width),          # 3. frame_shape
                    config,                   # 4. config
                    human_detection_results['human_mask'], # 5. human_mask (optional)
                    current_hand_positions    # 6. hand_positions (optional)
                    # num_segments, dynamic_history, human_segments, human_detection_history は渡さない
                )

            # !!! 注意: ここから下の dynamic_mask, dynamic_history, human_segments,
            # !!! human_detection_history の処理は、filter_dynamic_segments の
            # !!! 戻り値 (dynamic_indices) を使って pipeline.py 側で再実装する必要がある。
            # !!! 一旦、エラー解消のために dynamic_mask を仮生成する。
            dynamic_mask = np.zeros((height, width), dtype=np.uint8)
            if segmentation_mask is not None and dynamic_indices:
                 logger.debug(f"動的と判定されたセグメントインデックス: {dynamic_indices}")
                 # dynamic_indices に基づいて dynamic_mask を生成するロジックが必要
                 # segmentation_mask (IDマップ) と dynamic_indices を使う
                 for seg_idx in dynamic_indices:
                     if seg_idx + 1 >= 0: # IDは1から始まると仮定
                         dynamic_mask[segmentation_mask == (seg_idx + 1)] = 255
            # TODO: dynamic_history, human_segments, human_detection_history の更新ロジックもここに移動/再実装

            # デバッグ描画 (オプション)
            if config.save_debug_frames:
                # 元のフレームにマスクや情報を描画
                # 例: 動きマスクを半透明で重ねる
                motion_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                motion_colored[motion_mask > 0] = [0, 0, 255] # 赤色
                debug_frame = cv2.addWeighted(current_frame, 0.7, motion_colored, 0.3, 0)

                # 人間マスクを重ねる
                human_mask_colored = cv2.cvtColor(human_detection_results['human_mask'], cv2.COLOR_GRAY2BGR)
                human_mask_colored[human_detection_results['human_mask'] > 0] = [0, 255, 0] # 緑色
                debug_frame = cv2.addWeighted(debug_frame, 0.7, human_mask_colored, 0.3, 0)

                # 手の位置を描画
                for (hx, hy) in current_hand_positions:
                     cv2.circle(debug_frame, (int(hx), int(hy)), 5, (255, 0, 0), -1) # 青色
                # 前フレームの手の位置 (領域適応閾値用) も描画
                for (phx, phy) in prev_hand_positions:
                     cv2.circle(debug_frame, (int(phx), int(phy)), config.hand_region_radius, (255, 255, 0), 1) # 水色円

                # 動的と判断されたセグメントを強調表示
                final_dynamic_colored = np.zeros_like(debug_frame)
                final_dynamic_colored[dynamic_mask > 0] = [255, 0, 255] # マゼンタ
                debug_frame = cv2.addWeighted(debug_frame, 0.6, final_dynamic_colored, 0.4, 0)

                # フレーム番号等を描画
                utils.draw_text_with_background(debug_frame, f"Frame: {frame_idx}", (10, 30))

            # 6.8 結果の保存
            # マスク画像の保存 (動的部分が黒、静的部分が白)
            output_mask = cv2.bitwise_not(dynamic_mask)
            if config.save_masks:
                io.save_mask_image(output_mask, frame_idx, config, source_info)

            # デバッグフレームの保存
            if config.save_debug_frames:
                io.save_debug_frame(debug_frame, frame_idx, config, "debug")
                # 他のデバッグ情報 (motion_mask など) も保存可能
                # io.save_debug_frame(motion_mask, frame_idx, config, "motion")

            # 出力動画フレームの書き込み (マスク適用後 or デバッグフレーム)
            # final_output_frame = debug_frame # デバッグフレームを出力する場合
            # マスクを適用したフレームを出力する場合:
            final_output_frame = current_frame.copy()
            # final_output_frame[output_mask == 0] = [0, 0, 0] # 動的部分を黒塗り
            # または特定の背景色で塗りつぶすなど
            io.write_video_frame(video_writer, final_output_frame)

            # 6.9 状態の更新
            prev_frame = current_frame # 次のループのために現在のフレームを保持
            prev_result = segment_result # 次のループのためにセグメント結果を保持
            prev_frame_info = current_frame_info # 次のループのためにCOLMAP情報を保持
            prev_hand_positions = current_hand_positions # 次のループのために手の位置を保持

            # 処理時間計測とログ
            frame_end_time = time.time()
            proc_time_frame = frame_end_time - frame_start_time
            processing_times.append(proc_time_frame)
            avg_proc_time = sum(processing_times) / len(processing_times) if processing_times else 0
            current_fps = 1.0 / proc_time_frame if proc_time_frame > 0 else 0
            avg_fps = 1.0 / avg_proc_time if avg_proc_time > 0 else 0

            # --- [修正] 人間セグメントに属し、かつ動的なピクセル数を計算 --- #
            human_pixel_count = 0
            other_pixel_count = 0
            total_dynamic_pixel_count = np.count_nonzero(dynamic_mask)

            if human_segments and segmentation_mask is not None:
                 human_pixel_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
                 # segmentation_mask のデータ型とセグメントIDの範囲を確認する必要がある
                 # ここでは、human_segments に含まれるIDを持つピクセルをマークすると仮定
                 try:
                     # human_segments に非整数が含まれる可能性を考慮
                     valid_human_segment_ids = {int(s) for s in human_segments if isinstance(s, (int, float))}
                     for seg_id in valid_human_segment_ids:
                          if seg_id >= 0: # segmentation_mask の ID 形式に依存
                               human_pixel_mask[segmentation_mask == seg_id] = 255

                     # 人間ピクセルマスクと動的マスクの共通部分を計算
                     human_dynamic_intersection = cv2.bitwise_and(human_pixel_mask, dynamic_mask)
                     human_pixel_count = np.count_nonzero(human_dynamic_intersection)

                 except Exception as e_intersect:
                      logger.warning(f"人間/動的ピクセル数の計算中にエラー: {e_intersect}", exc_info=False)

            other_pixel_count = total_dynamic_pixel_count - human_pixel_count
            # --- 修正ここまで --- #

            # テキスト描画 (ピクセル数表示に変更)
            utils.draw_text_with_background(final_output_frame, f"Dyn Pixels: {other_pixel_count} / Hum Pixels: {human_pixel_count}", (10, 30), text_color=(0, 0, 255), font_scale=0.7)
            utils.draw_text_with_background(final_output_frame, f"Time: {proc_time_frame:.3f}s FPS: {current_fps:.1f}", (10, height - 30), text_color=(0, 255, 0), font_scale=0.6)
            utils.draw_text_with_background(final_output_frame, f"Avg Time: {avg_proc_time:.3f}s Avg FPS: {avg_fps:.1f}", (10, height - 10), text_color=(0, 255, 0), font_scale=0.6)

            # 動画フレーム書き込み
            io.write_video_frame(video_writer, final_output_frame)

            # デバッグフレーム保存 (一定間隔)
            if frame_idx % int(fps if fps > 0 else 30) == 0:
                 io.save_debug_frame(final_output_frame, frame_idx, config, "frame")
            if frame_idx % 10 == 0:
                 io.save_debug_frame(final_output_frame, frame_idx, config, "time_frame")

            # --- ループ終端処理 --- #
            frame_idx += 1

            # ログ出力 (一定間隔)
            if frame_idx % 10 == 0:
                logger.info(f"フレーム {frame_idx} 処理完了... Time: {proc_time_frame:.3f}s, FPS: {current_fps:.1f}, Avg FPS: {avg_fps:.1f}")

            # (オプション) cv2.imshow で表示
            # if config.show_frames:
            #     cv2.imshow("DynaMask Output", final_output_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         logger.info("'q'キーが押されたため、処理を中断します。")
            #         break

        # --- ループ終了 --- #

    except KeyboardInterrupt:
        logger.info("キーボード割り込みにより処理が中断されました。")
    except Exception as e:
        logger.critical(f"フレーム処理ループ中に予期せぬエラーが発生しました: {e}", exc_info=True)
    finally:
        # --- リソース解放 --- #
        logger.info("リソースを解放しています...")
        executor.shutdown(wait=True) # スレッドプールを閉じる
        io.release_video_writer(video_writer)

        # 入力ソースの解放 (video_segment に依存)
        if source_info and hasattr(video_segment, 'release_input_source'):
            try:
                video_segment.release_input_source(source_info)
                logger.info("入力ソースを解放しました。")
            except Exception as e:
                logger.error(f"入力ソースの解放中にエラー: {e}", exc_info=True)
        elif source_info and 'cap' in source_info and hasattr(source_info['cap'], 'release'):
             try:
                 source_info['cap'].release()
                 logger.info("入力ソース (cap) を解放しました。")
             except Exception as e:
                 logger.error(f"入力ソース (cap) の解放中にエラー: {e}", exc_info=True)

        # OpenCVウィンドウ (もし表示していた場合)
        # if config.show_frames:
        #     try:
        #         cv2.destroyAllWindows()
        #     except Exception:
        #         pass # エラーは無視

        # 処理時間統計
        end_pipeline_time = time.time()
        total_pipeline_time = end_pipeline_time - start_pipeline_time
        if processing_times:
            avg_proc_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_proc_time if avg_proc_time > 0 else 0
            min_time = min(processing_times)
            max_time = max(processing_times)
            min_fps = 1.0 / max_time if max_time > 0 else 0
            max_fps = 1.0 / min_time if min_time > 0 else 0

            logger.info(f"===== 処理時間統計 =====")
            logger.info(f"平均処理時間: {avg_proc_time:.4f}秒/フレーム (平均FPS: {avg_fps:.2f})")
            logger.info(f"最短処理時間: {min_time:.4f}秒/フレーム (最大FPS: {max_fps:.2f})")
            logger.info(f"最長処理時間: {max_time:.4f}秒/フレーム (最小FPS: {min_fps:.2f})")
            logger.info(f"総処理時間: {total_pipeline_time:.2f}秒, 総フレーム数: {frame_idx}")
        else:
            logger.info("フレーム処理が実行されなかったため、時間統計はありません。")

        logger.info(f"DynaMask パイプライン完了。総時間: {total_pipeline_time:.2f} 秒")
        logger.info(f"出力はディレクトリ '{config.output_dir}' に保存されました。")

        return config.output_dir # 成功時は出力ディレクトリを返す

# このファイルが直接実行された場合の処理 (デバッグ用など)
# if __name__ == '__main__':
#     # ここで設定を読み込み、run_dynamask を呼び出す
#     # 例: config = DynaMaskConfig(...) # デフォルト設定や引数解析
#     # logging.basicConfig(level=logging.INFO) # 基本的なロギング設定
#     # run_dynamask(config)
#     pass

