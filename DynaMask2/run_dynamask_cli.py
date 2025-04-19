"""
DynaMask2 コマンドラインインターフェース
"""
import argparse
import logging
import os
import sys

# dynamask パッケージをインポート可能にするため、
# スクリプトの親ディレクトリ (DynaMask2) をパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# 内部モジュールインポート (パス追加後に行う)
from dynamask.config import DynaMaskConfig
from dynamask.pipeline import run_dynamask
from dynamask.utils import setup_logging


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(
        description="DynaMask2: カメラ移動を考慮した動的要素マスキングツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- 入力設定 ---
    input_group = parser.add_argument_group('入力設定')
    input_source_group = input_group.add_mutually_exclusive_group(required=True)
    input_source_group.add_argument("--video", type=str, help="入力動画ファイルのパス")
    input_source_group.add_argument("--images", type=str, help="入力画像シーケンスのフォルダパス")
    input_source_group.add_argument("--camera", type=int, help="カメラデバイスID（指定するとカメラ入力モード）")
    input_group.add_argument("--image-pattern", type=str, default="*.png",
                           help="画像シーケンスのファイルパターン")

    # --- 出力設定 ---
    output_group = parser.add_argument_group('出力設定')
    output_group.add_argument("--output", type=str, default=None,
                            dest='output_dir',
                            help="出力ディレクトリ (デフォルト: ./output/dyna_YYYYMMDD_HHMMSS)")
    output_group.add_argument("--output-filename", type=str, default="dynamic_masked.mp4",
                            help="出力動画ファイル名")
    output_group.add_argument("--save-debug", action="store_true",
                            dest='save_debug_frames',
                            help="デバッグ用の中間結果フレームを保存する")
    output_group.add_argument("--save-masks", action="store_true",
                            help="最終的な動的マスク画像（白背景/黒マスク）を保存する")
    # output_group.add_argument("--no-show", dest="show", action="store_false", help="フレームを表示しない") # 表示機能は pipeline から削除されたため不要

    # --- FastSAM設定 ---
    fastsam_group = parser.add_argument_group('FastSAM 設定')
    fastsam_group.add_argument("--fastsam-model", type=str, default="FastSAM-x",
                               help="使用するFastSAMモデル名 (例: FastSAM-s, FastSAM-x)")
    fastsam_group.add_argument("--fastsam-conf", type=float, default=0.7,
                               help="FastSAMセグメンテーションの信頼度閾値")
    fastsam_group.add_argument("--imgsz", type=int, default=1024,
                               help="FastSAMへの入力画像サイズ")

    # --- 動き検出設定 ---
    motion_group = parser.add_argument_group('動き検出設定')
    motion_group.add_argument("--motion-threshold", type=float, default=80.0,
                             help="オプティカルフローの動きマグニチュード閾値スケール")
    motion_group.add_argument("--no-camera-compensation", dest="camera_compensation",
                               action="store_false", default=True,
                               help="オプティカルフローによるカメラ動き補正を無効化")
    motion_group.add_argument("--optical-flow-winsize", type=int, default=15,
                               help="オプティカルフロー計算のウィンドウサイズ")
    motion_group.add_argument("--motion-history", type=int, default=3,
                            dest='motion_history_frames',
                           help="動きマスクの時間的安定化に使用する履歴フレーム数")

    # --- [New!] COLMAP 自己運動分離設定 --- #
    colmap_group = parser.add_argument_group('COLMAP 自己運動分離設定')
    colmap_group.add_argument("--colmap_images_path", type=str, default=None,
                              help="COLMAPの images.txt ファイルへのパス")
    colmap_group.add_argument("--colmap_cameras_path", type=str, default=None,
                              help="COLMAPの cameras.txt ファイルへのパス")
    colmap_group.add_argument("--use_colmap_egomotion", action=argparse.BooleanOptionalAction, default=True,
                             help="COLMAPポーズ情報が存在する場合に自己運動分離を試みる (--no-use_colmap_egomotion で無効化)")
    colmap_group.add_argument("--fallback_to_flow_compensation", action=argparse.BooleanOptionalAction, default=True,
                             help="COLMAP分離失敗/無効時にオプティカルフロー補正を行うか (--no-fallback_to_flow_compensation で無効化)")

    # --- [New!] 領域適応閾値設定 --- #
    adaptive_thresh_group = parser.add_argument_group('領域適応閾値設定')
    adaptive_thresh_group.add_argument("--use_region_adaptive_threshold", action=argparse.BooleanOptionalAction, default=True,
                                       help="手の検出時に領域適応閾値を有効にするか (--no-use_region_adaptive_threshold で無効化)")
    adaptive_thresh_group.add_argument("--hand_region_radius", type=int, default=100,
                                       help="手の位置を中心とした作業領域の半径 (ピクセル)")
    adaptive_thresh_group.add_argument("--hand_region_threshold_factor", type=float, default=0.5,
                                       help="背景閾値に対する作業領域閾値の比率 (小さいほど敏感)")

    # --- セグメント判定設定 ---
    segment_group = parser.add_argument_group('セグメント判定設定')
    segment_group.add_argument("--min-area", dest="min_area_ratio", type=float, default=0.01,
                              help="動的候補とするセグメント面積の最小比率 (対画像サイズ)")
    segment_group.add_argument("--max-area", dest="max_area_ratio", type=float, default=0.25,
                              help="動的候補とするセグメント面積の最大比率")
    segment_group.add_argument("--temporal", dest="temporal_consistency", type=int, default=3,
                              help="動的と最終判定するのに必要な連続検出フレーム数")
    segment_group.add_argument("--motion-overlap-ratio", type=float, default=0.45,
                              help="セグメントが動的と判定される動きマスクとの最小重なり率")
    segment_group.add_argument("--min-motion-pixels", type=int, default=100,
                              help="セグメント内の動きとして考慮する最小ピクセル数")
    segment_group.add_argument("--center-motion-ratio", type=float, default=0.35,
                              help="セグメント中心領域の動き検出に必要な重なり率")
    segment_group.add_argument("--dynamic-decay", dest="dynamic_decay_rate", type=float, default=0.7,
                           help="動的カウントの減衰率 (小さいほど判定が持続)")
    segment_group.add_argument("--temporal-offset", dest="temporal_threshold_offset", type=float, default=0.8,
                           help="動的判定の最終閾値へのオフセット (大きいほど厳格)")

    # --- 人間検出設定 ---
    human_group = parser.add_argument_group('人間検出設定')
    human_group.add_argument("--no-pose", dest="use_pose_detection", action="store_false", default=True,
                            help="MediaPipeによるポーズ/手検出を無効化")
    human_group.add_argument("--pose-confidence", type=float, default=0.5,
                            help="MediaPipeポーズ検出の信頼度閾値")
    human_group.add_argument("--hand-confidence", type=float, default=0.5,
                            help="MediaPipe手検出の信頼度閾値")
    human_group.add_argument("--hand-proximity", dest="hand_proximity_threshold", type=int, default=50,
                            help="セグメントが手の近傍と判定される最大距離 (pixel)")
    human_group.add_argument("--hand-proximity-factor", type=float, default=0.7,
                            help="手の近傍にあるセグメントの動的判定緩和係数 (1.0で緩和なし)")
    human_group.add_argument("--no-yolo", dest="use_yolo", action="store_false", default=True,
                            help="YOLOv8による人間検出を無効化")
    human_group.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                            help="使用するYOLOv8モデル名")
    human_group.add_argument("--yolo-confidence", type=float, default=0.4,
                            help="YOLOv8人間検出の信頼度閾値")
    human_group.add_argument("--human-overlap", dest="human_overlap_ratio", type=float, default=0.4,
                            help="セグメントが人間と判定されるマスクとの最小重なり率")

    # --- その他 --- # 
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="ログレベルを設定")
    parser.add_argument("--log-file", type=str, default=None,
                           help="ログをファイルに出力する場合のパス")

    return parser.parse_args()

def main():
    """メイン実行関数"""
    args = parse_arguments()

    # ロギング設定
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO), log_file=args.log_file)

    # 入力タイプ決定
    input_type = None
    input_path = None
    if args.video:
        input_type = "video"
        input_path = args.video
    elif args.images:
        input_type = "images"
        input_path = args.images
    elif args.camera is not None:
        input_type = "camera"
        input_path = str(args.camera) # パスとしてカメラIDを渡す（pipeline内で処理）
    else:
        # argparse required=True で通常は到達しないはず
        logging.critical("入力ソース (--video, --images, --camera) が指定されていません。")
        sys.exit(1)

    # 設定オブジェクト作成 (parse_argsの結果を**vars()で展開して渡すことで、追加した引数も自動的に反映させる)
    try:
        # args 名前空間の属性名を config のフィールド名と一致させる必要がある
        # 一部の引数名を手動で調整 (--output -> output_dir, --save-debug -> save_debug_frames など)
        config_kwargs = vars(args).copy() # コピーして操作

        # 入力ソース情報を kwargs に追加 (DynaMaskConfigのフィールド名に合わせる)
        config_kwargs['input_type'] = input_type
        config_kwargs['input_path'] = input_path

        # 不要になった引数や名前が違う引数を削除 (args から直接渡さないもの)
        # config_kwargs.pop('video', None)
        # config_kwargs.pop('images', None)
        # config_kwargs.pop('camera', None)
        # config_kwargs.pop('log_level', None) # ロギング設定は Config 外
        # config_kwargs.pop('log_file', None)

        # 手動で名前を合わせたキーがあれば不要だが、念のためチェック
        if 'output' in config_kwargs and 'output_dir' not in config_kwargs:
             config_kwargs['output_dir'] = config_kwargs.pop('output')
        if 'save_debug' in config_kwargs and 'save_debug_frames' not in config_kwargs:
             config_kwargs['save_debug_frames'] = config_kwargs.pop('save_debug')
        if 'temporal' in config_kwargs and 'temporal_consistency' not in config_kwargs:
             config_kwargs['temporal_consistency'] = config_kwargs.pop('temporal')
        if 'min_area' in config_kwargs and 'min_area_ratio' not in config_kwargs:
             config_kwargs['min_area_ratio'] = config_kwargs.pop('min_area')
        if 'max_area' in config_kwargs and 'max_area_ratio' not in config_kwargs:
             config_kwargs['max_area_ratio'] = config_kwargs.pop('max_area')
        if 'no_pose' in config_kwargs:
             config_kwargs['use_pose_detection'] = not config_kwargs.pop('no_pose') # BooleanOptionalActionを使ったので不要かも
        if 'no_yolo' in config_kwargs:
             config_kwargs['use_yolo'] = not config_kwargs.pop('no_yolo') # BooleanOptionalActionを使ったので不要かも
        if 'no_camera_compensation' in config_kwargs:
             config_kwargs['camera_compensation'] = not config_kwargs.pop('no_camera_compensation') # BooleanOptionalActionを使ったので不要かも
        if 'motion_history' in config_kwargs and 'motion_history_frames' not in config_kwargs:
             config_kwargs['motion_history_frames'] = config_kwargs.pop('motion_history')
        if 'fastsam_conf' in config_kwargs and 'fastsam_confidence' not in config_kwargs:
             config_kwargs['fastsam_confidence'] = config_kwargs.pop('fastsam_conf')
        if 'fastsam_model' in config_kwargs and 'fastsam_model_name' not in config_kwargs:
             config_kwargs['fastsam_model_name'] = config_kwargs.pop('fastsam_model')
        if 'hand_proximity' in config_kwargs and 'hand_proximity_threshold' not in config_kwargs:
             config_kwargs['hand_proximity_threshold'] = config_kwargs.pop('hand_proximity')
        if 'human_overlap' in config_kwargs and 'human_overlap_ratio' not in config_kwargs:
             config_kwargs['human_overlap_ratio'] = config_kwargs.pop('human_overlap')
        if 'dynamic_decay' in config_kwargs and 'dynamic_decay_rate' not in config_kwargs:
             config_kwargs['dynamic_decay_rate'] = config_kwargs.pop('dynamic_decay')
        if 'temporal_offset' in config_kwargs and 'temporal_threshold_offset' not in config_kwargs:
             config_kwargs['temporal_threshold_offset'] = config_kwargs.pop('temporal_offset')


        # 不要なキーを削除 (DynaMaskConfigに存在しないもの)
        allowed_keys = DynaMaskConfig.__annotations__.keys()
        keys_to_remove = [k for k in config_kwargs if k not in allowed_keys]
        for key in keys_to_remove:
            # logger.debug(f"Config生成時に不要な引数を削除: {key}={config_kwargs[key]}")
            config_kwargs.pop(key)


        # Configオブジェクトを生成
        config = DynaMaskConfig(**config_kwargs)

    except Exception as e:
        logging.critical(f"設定オブジェクトの作成中にエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

    # パイプライン実行
    try:
        logging.info("=" * 60)
        logging.info(" DynaMask2 処理開始")
        logging.info("=" * 60)
        result_dir = run_dynamask(config)
        if result_dir:
            logging.info("=" * 60)
            logging.info(f"処理が正常に完了しました。出力: {result_dir}")
            logging.info("=" * 60)
        else:
            logging.error("処理中にエラーが発生しました。詳細はログを確認してください。")
            sys.exit(1)
    except Exception as e:
        logging.critical(f"パイプライン実行中に予期せぬエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 