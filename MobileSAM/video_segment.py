"""
MobileSAM Video Segmentation Script

動画、画像シーケンス、カメラ入力に対して MobileSAM モデルを使用したセマンティックセグメンテーションを行います。
"""

import cv2
import numpy as np
import torch
import os
import glob
import sys
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Union, Any, Tuple

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MobileSAM")

# Ultralyticsを使ってSAMをインポート
try:
    from ultralytics import SAM

    USING_ULTRALYTICS = True
except ImportError:
    USING_ULTRALYTICS = False


@dataclass
class Config:
    """MobileSAMを使用した動画/画像セグメンテーションの設定クラス"""

    # 入力設定
    input_type: str = "video"  # "video", "images", "camera"のいずれか
    input_path: Optional[str] = None  # 動画ファイルパスまたは画像フォルダパス
    camera_id: int = 0  # カメラデバイスID
    image_pattern: str = "*.jpg"  # 画像シーケンスのパターン

    # モデル設定
    model_path: str = "mobile_sam.pt"  # モデルファイルパス
    device: Optional[str] = None  # 処理デバイス (cuda/cpu、Noneなら自動検出)

    # セグメンテーション設定
    points: Optional[list] = None  # セグメントするポイント
    boxes: Optional[list] = None  # セグメントするボックス
    auto_everything: bool = False  # すべてのオブジェクトを自動セグメントするか
    labels: Optional[list] = None  # ポイントのラベル（1=正、0=負）
    imgsz: Optional[int] = None  # 入力画像サイズ (Noneなら自動設定)
    auto_size: bool = True  # 動画サイズに自動調整するか

    # 出力設定
    save_frames: bool = True  # フレームを保存するか
    show_frames: bool = True  # フレームを表示するか
    frame_save_interval: int = 30  # 何フレームごとに画像を保存するか
    display_scale: float = 0.5  # 表示時のスケール (1.0で原寸大)

    # ファイル設定
    output_dir: Optional[str] = None  # 出力ディレクトリ (Noneなら自動生成)
    output_filename: str = "segmented_output.mp4"  # 出力動画ファイル名

    # カメラ設定
    camera_fps: float = 30.0  # カメラ入力時のFPS
    camera_width: int = 1280  # カメラ解像度 (幅)
    camera_height: int = 720  # カメラ解像度 (高さ)

    # 表示設定
    mask_alpha: float = 0.4  # マスクの透明度 (0.0-1.0)
    show_labels: bool = True  # セグメント数などのラベルを表示するか
    use_random_colors: bool = True  # セグメントにランダムな色を使用するか
    overlay_mask: bool = True  # マスクをオーバーレイ表示するか
    draw_contours: bool = True  # マスクの輪郭を描画するか

    def __post_init__(self) -> None:
        """設定の初期化後の処理"""
        # 出力ディレクトリの設定
        if self.output_dir is None:
            # outputフォルダの中に日時フォルダを作成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("output", timestamp)

        # デバイスの自動検出
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 入力サイズの設定
        if self.imgsz is not None:
            self.auto_size = False

        # points、boxes、またはauto_everythingの少なくとも1つが指定されているか確認
        if not self.auto_everything and self.points is None and self.boxes is None:
            # デフォルトではランダムな点を1つ選択
            logger.info("ポイントが指定されていないため、画像中央の点を使用します")
            self.points = None  # 実行時に画像サイズに基づいて設定

        # labelsのデフォルト設定
        if self.points is not None and self.labels is None:
            self.labels = [1] * (
                len(self.points) if isinstance(self.points[0], list) else 1
            )


def load_model(config: Config) -> Any:
    """MobileSAMモデルをロードする関数

    Args:
        config: 設定オブジェクト

    Returns:
        ロードされたモデル
    """
    if USING_ULTRALYTICS:
        # ultralyticsパッケージを使用
        model = SAM(config.model_path)
    else:
        raise ImportError(
            "ultralytics パッケージがインストールされていません。"
            "`pip install ultralytics` を実行してインストールしてください。"
        )

    # GPU設定
    if config.device == "cuda" and torch.cuda.is_available():
        # GPUを使用
        pass
    else:
        config.device = "cpu"

    return model


def initialize_input_source(config: Config) -> Dict[str, Any]:
    """入力ソース（動画、画像シーケンス、カメラ）を初期化する関数

    Args:
        config: 設定オブジェクト

    Returns:
        入力ソース情報を含む辞書
    """
    width = None
    height = None
    fps = 30.0
    frame_count = 0
    cap = None
    image_files = []

    if config.input_type == "video":
        if not config.input_path:
            raise ValueError("動画入力モードでは入力パスが必要です")

        cap = cv2.VideoCapture(config.input_path)
        if not cap.isOpened():
            raise ValueError(f"動画 {config.input_path} を開けませんでした")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    elif config.input_type == "images":
        if not config.input_path:
            raise ValueError("画像シーケンス入力モードでは入力フォルダが必要です")

        search_path = os.path.join(config.input_path, config.image_pattern)
        image_files = sorted(glob.glob(search_path))

        if not image_files:
            raise ValueError(
                f"指定されたパターン {search_path} に一致する画像がありません"
            )

        sample_img = cv2.imread(image_files[0])
        if sample_img is None:
            raise ValueError(f"画像 {image_files[0]} を読み込めませんでした")

        height, width = sample_img.shape[:2]
        frame_count = len(image_files)

    elif config.input_type == "camera":
        cap = cv2.VideoCapture(config.camera_id)
        if not cap.isOpened():
            raise ValueError(f"カメラID {config.camera_id} を開けませんでした")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = config.camera_fps
        frame_count = float("inf")  # カメラ入力は無限フレーム

    else:
        raise ValueError(f"未対応の入力タイプ: {config.input_type}")

    return {
        "cap": cap,
        "image_files": image_files,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def get_next_frame(
    source: Dict[str, Any], frame_idx: int
) -> Tuple[bool, Optional[np.ndarray]]:
    """入力ソースから次のフレームを取得する関数

    Args:
        source: 入力ソース情報
        frame_idx: 現在のフレームインデックス

    Returns:
        (成功フラグ, フレーム画像)のタプル
    """
    frame = None
    ret = False

    if source["cap"] is not None:
        # 動画またはカメラからフレームを取得
        ret, frame = source["cap"].read()
    elif 0 <= frame_idx < len(source["image_files"]):
        # 画像シーケンスからフレームを取得
        image_path = source["image_files"][frame_idx]
        frame = cv2.imread(image_path)
        ret = frame is not None
    else:
        ret = False

    return ret, frame


def predict_with_model(model: Any, frame: np.ndarray, config: Config) -> Any:
    """MobileSAMモデルで予測を行う関数

    Args:
        model: ロードされたモデル
        frame: 入力フレーム
        config: 設定オブジェクト

    Returns:
        予測結果
    """
    # ポイントプロンプトが未設定の場合、画像中央の点を使用
    if config.points is None and not config.auto_everything and config.boxes is None:
        h, w = frame.shape[:2]
        config.points = [w // 2, h // 2]  # 画像中央の点

    # 予測実行
    if config.auto_everything:
        # すべてのオブジェクトをセグメント
        results = model.predict(source=frame, device=config.device)
    elif config.boxes is not None:
        # ボックスプロンプトでセグメント
        results = model.predict(
            source=frame,
            device=config.device,
            boxes=config.boxes,
        )
    else:
        # ポイントプロンプトでセグメント
        results = model.predict(
            source=frame,
            device=config.device,
            points=config.points,
            labels=config.labels,
        )

    return results


def process_results(results: Any, frame: np.ndarray, config: Config) -> np.ndarray:
    """MobileSAMの予測結果を処理してビジュアライズする関数

    Args:
        results: モデルの予測結果
        frame: 元の入力フレーム
        config: 設定オブジェクト

    Returns:
        可視化されたフレーム
    """
    # 結果の可視化
    if hasattr(results, "plot"):
        # ultralyticsのplot関数を使用
        return results[0].plot()
    else:
        # 独自の可視化処理
        h, w = frame.shape[:2]
        result_frame = frame.copy()

        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for i, mask in enumerate(result.masks.data):
                    # マスクをnumpy配列に変換して正規化
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    mask_np = cv2.resize(mask_np, (w, h))

                    # ランダムな色を生成（または固定色）
                    if config.use_random_colors:
                        color = np.random.randint(
                            0, 255, size=3, dtype=np.uint8
                        ).tolist()
                    else:
                        color = [0, 255, 0]  # 緑色をデフォルトに

                    # マスクをオーバーレイ
                    if config.overlay_mask:
                        colored_mask = np.zeros_like(frame, dtype=np.uint8)
                        colored_mask[mask_np > 0] = color

                        # アルファブレンディング
                        cv2.addWeighted(
                            colored_mask,
                            config.mask_alpha,
                            result_frame,
                            1 - config.mask_alpha,
                            0,
                            result_frame,
                        )

                    # 輪郭を描画
                    if config.draw_contours:
                        contours, _ = cv2.findContours(
                            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(result_frame, contours, -1, color, 2)

        # セグメント数を表示
        if config.show_labels:
            segment_count = (
                len(results[0].masks.data)
                if hasattr(results[0], "masks") and results[0].masks is not None
                else 0
            )
            cv2.putText(
                result_frame,
                f"Segments: {segment_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        return result_frame


def process_video(input_path: Union[str, int], config: Optional[Config] = None) -> str:
    """動画をMobileSAMで処理する関数

    Args:
        input_path: 入力動画パスまたはカメラID
        config: 設定オブジェクト（指定なければデフォルト設定で作成）

    Returns:
        出力動画パス
    """
    if config is None:
        config = Config()

    if isinstance(input_path, str):
        config.input_path = input_path
        config.input_type = "video"
    else:
        config.camera_id = input_path
        config.input_type = "camera"

    # 出力ディレクトリの作成
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"出力ディレクトリ: {config.output_dir}")

    # モデルのロード
    logger.info(f"MobileSAMモデルをロード中: {config.model_path}")
    model = load_model(config)
    logger.info(f"使用デバイス: {config.device}")

    # 入力ソースの初期化
    logger.info(f"入力を初期化中: {config.input_type}")
    source = initialize_input_source(config)
    width, height = source["width"], source["height"]
    fps = source["fps"]
    logger.info(f"入力サイズ: {width}x{height}, FPS: {fps:.2f}")

    # 出力動画ライターの設定
    output_video_path = os.path.join(config.output_dir, config.output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = source["frame_count"]
    if total_frames < float("inf"):
        logger.info(f"処理するフレーム数: {total_frames}")

    try:
        frame_idx = 0
        while True:
            # フレームの読み込み
            ret, frame = get_next_frame(source, frame_idx)
            if not ret:
                break

            if frame_idx % 10 == 0:
                logger.info(f"フレーム {frame_idx} を処理中...")

            # モデルで予測
            results = predict_with_model(model, frame, config)

            # 結果を処理して可視化
            result_frame = process_results(results, frame, config)

            # フレームの保存
            out.write(result_frame)

            # 指定したフレーム間隔で画像を保存
            if config.save_frames and frame_idx % config.frame_save_interval == 0:
                frame_save_path = os.path.join(
                    config.output_dir, f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(frame_save_path, result_frame)

            # フレームの表示
            if config.show_frames:
                if config.display_scale != 1.0:
                    display_h = int(height * config.display_scale)
                    display_w = int(width * config.display_scale)
                    display_frame = cv2.resize(result_frame, (display_w, display_h))
                else:
                    display_frame = result_frame

                cv2.imshow("MobileSAM Segmentation", display_frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESCキーで終了
                    break

            frame_idx += 1

    finally:
        # リソースの解放
        if source["cap"] is not None:
            source["cap"].release()
        out.release()
        cv2.destroyAllWindows()

    logger.info(f"{frame_idx} フレームを処理しました")
    logger.info(f"出力動画保存先: {output_video_path}")

    return output_video_path


def parse_arguments():
    """コマンドライン引数を解析する関数

    Returns:
        設定オブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(description="MobileSAM 動画セグメンテーション")

    # 入力関連
    input_group = parser.add_argument_group("入力設定")
    input_sources = input_group.add_mutually_exclusive_group(required=True)
    input_sources.add_argument("--video", type=str, help="入力動画のパス")
    input_sources.add_argument("--images", type=str, help="入力画像のディレクトリパス")
    input_sources.add_argument("--camera", type=int, help="カメラデバイスID", default=0)
    input_group.add_argument(
        "--image-pattern",
        type=str,
        default="*.jpg",
        help="画像ファイルのパターン (例: '*.jpg')",
    )

    # モデル関連
    model_group = parser.add_argument_group("モデル設定")
    model_group.add_argument(
        "--model", type=str, default="mobile_sam.pt", help="MobileSAMモデルのパス"
    )
    model_group.add_argument(
        "--device", type=str, help="使用するデバイス ('cuda' または 'cpu')"
    )

    # セグメンテーション関連
    seg_group = parser.add_argument_group("セグメンテーション設定")
    seg_group.add_argument(
        "--points",
        type=float,
        nargs="+",
        help="セグメントするポイント座標 (x1 y1 x2 y2 ...)",
    )
    seg_group.add_argument(
        "--boxes",
        type=float,
        nargs="+",
        help="セグメントするボックス座標 (x1 y1 x2 y2 ...)",
    )
    seg_group.add_argument(
        "--labels",
        type=int,
        nargs="+",
        help="ポイントのラベル (1=ポジティブ, 0=ネガティブ)",
    )
    seg_group.add_argument(
        "--everything", action="store_true", help="すべてのオブジェクトをセグメント"
    )
    seg_group.add_argument("--imgsz", type=int, help="入力画像サイズ")

    # 出力関連
    output_group = parser.add_argument_group("出力設定")
    output_group.add_argument("--output", type=str, help="出力ディレクトリ")
    output_group.add_argument(
        "--output-video",
        type=str,
        default="segmented_output.mp4",
        help="出力動画ファイル名",
    )
    output_group.add_argument(
        "--no-save", action="store_true", help="フレームを保存しない"
    )
    output_group.add_argument(
        "--no-show", action="store_true", help="リアルタイムでフレームを表示しない"
    )
    output_group.add_argument(
        "--frame-interval", type=int, default=30, help="フレーム保存間隔"
    )
    output_group.add_argument(
        "--display-scale", type=float, default=0.5, help="表示スケール"
    )

    # 表示関連
    view_group = parser.add_argument_group("表示設定")
    view_group.add_argument(
        "--mask-alpha", type=float, default=0.4, help="マスクの透明度 (0.0-1.0)"
    )
    view_group.add_argument(
        "--no-labels", action="store_true", help="ラベル表示を無効化"
    )
    view_group.add_argument(
        "--fixed-color", action="store_true", help="固定色を使用 (ランダム色を無効化)"
    )
    view_group.add_argument(
        "--no-overlay", action="store_true", help="マスクオーバーレイを無効化"
    )
    view_group.add_argument(
        "--no-contours", action="store_true", help="輪郭描画を無効化"
    )

    args = parser.parse_args()

    # Config オブジェクトの作成
    config = Config()

    # 入力設定
    if args.video:
        config.input_type = "video"
        config.input_path = args.video
    elif args.images:
        config.input_type = "images"
        config.input_path = args.images
        config.image_pattern = args.image_pattern
    elif args.camera is not None:
        config.input_type = "camera"
        config.camera_id = args.camera

    # モデル設定
    if args.model:
        config.model_path = args.model
    if args.device:
        config.device = args.device

    # セグメンテーション設定
    if args.points:
        # [x1, y1, x2, y2, ...] を [[x1, y1], [x2, y2], ...] に変換
        points = args.points
        config.points = []
        for i in range(0, len(points), 2):
            if i + 1 < len(points):
                config.points.append([points[i], points[i + 1]])

    if args.boxes:
        # [x1, y1, x2, y2, ...] を [[x1, y1, x2, y2], ...] に変換
        boxes = args.boxes
        config.boxes = []
        for i in range(0, len(boxes), 4):
            if i + 3 < len(boxes):
                config.boxes.append(boxes[i : i + 4])

    if args.labels:
        config.labels = args.labels

    if args.everything:
        config.auto_everything = True

    if args.imgsz:
        config.imgsz = args.imgsz

    # 出力設定
    if args.output:
        config.output_dir = args.output
    if args.output_video:
        config.output_filename = args.output_video
    if args.no_save:
        config.save_frames = False
    if args.no_show:
        config.show_frames = False
    if args.frame_interval:
        config.frame_save_interval = args.frame_interval
    if args.display_scale:
        config.display_scale = args.display_scale

    # 表示設定
    if args.mask_alpha:
        config.mask_alpha = args.mask_alpha
    if args.no_labels:
        config.show_labels = False
    if args.fixed_color:
        config.use_random_colors = False
    if args.no_overlay:
        config.overlay_mask = False
    if args.no_contours:
        config.draw_contours = False

    return config


def main():
    """メイン関数"""
    config = parse_arguments()

    logger.info("MobileSAM 動画セグメンテーションを開始...")

    try:
        if config.input_type == "video":
            process_video(config.input_path, config)
        elif config.input_type == "images":
            # 画像シーケンス処理
            logger.info(f"画像シーケンス処理: {config.input_path}")
            process_video(config.input_path, config)
        elif config.input_type == "camera":
            # カメラ処理
            logger.info(f"カメラ処理: ID {config.camera_id}")
            process_video(config.camera_id, config)
    except KeyboardInterrupt:
        logger.info("ユーザーによる処理中断")
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return 1

    logger.info("処理完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
