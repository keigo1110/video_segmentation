"""
FastSAM Video Segmentation Script

動画、画像シーケンス、カメラ入力に対して FastSAM モデルを使用したセマンティックセグメンテーションを行います。
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
logger = logging.getLogger("FastSAM")

# FastSAMのインポート
try:
    from ultralytics import FastSAM

    USING_ULTRALYTICS = True
except ImportError:
    USING_ULTRALYTICS = False


@dataclass
class Config:
    """FastSAMを使用した動画/画像セグメンテーションの設定クラス"""

    # 入力設定
    input_type: str = "video"  # "video", "images", "camera"のいずれか
    input_path: Optional[str] = None  # 動画ファイルパスまたは画像フォルダパス
    camera_id: int = 0  # カメラデバイスID
    image_pattern: str = "*.jpg"  # 画像シーケンスのパターン

    # モデル設定
    model_name: str = "FastSAM-x"  # "FastSAM-s" または "FastSAM-x"
    model_path: Optional[str] = None  # モデルファイルパス
    device: Optional[str] = None  # 処理デバイス (cuda/cpu、Noneなら自動検出)

    # セグメンテーション設定
    confidence: float = 0.7  # 検出の信頼度しきい値
    iou: float = 0.7  # IOU (Intersection over Union) しきい値
    retina_masks: bool = True  # 高品質マスクを使用するか
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

        # FastSAMモデルパスの設定
        if not USING_ULTRALYTICS and self.model_path is None:
            # モデルファイルを探索する順序：
            # 1. カレントディレクトリのweightsフォルダ
            # 2. 親ディレクトリのFastSAM/weightsフォルダ
            model_filename = f"{self.model_name}.pt"

            # weightsディレクトリを確認
            weights_dir = "weights"
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)

            self.model_path = os.path.join(weights_dir, model_filename)

            # ファイルが存在しない場合は代替パスをチェック
            if not os.path.exists(self.model_path):
                alt_path = os.path.join("../FastSAM/weights", model_filename)
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                else:
                    # どちらのパスにもモデルが見つからない場合は警告
                    logger.warning(
                        f"モデルファイル {model_filename} が見つかりません。"
                        f"以下のいずれかの場所にモデルファイルを配置してください:\n"
                        f"- {os.path.abspath(self.model_path)}\n"
                        f"- {os.path.abspath(alt_path)}"
                    )


def load_model(config: Config) -> Any:
    """FastSAMモデルをロードする関数

    Args:
        config: 設定オブジェクト

    Returns:
        ロードされたモデル
    """
    if USING_ULTRALYTICS:
        # ultralyticsパッケージを使用
        model = FastSAM(config.model_name)
    else:
        # FastSAMリポジトリを直接使用
        fastsam_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../FastSAM"
        )
        sys.path.append(fastsam_path)
        from fastsam import FastSAM as OriginalFastSAM

        model = OriginalFastSAM(config.model_path)

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

    return ret, frame


def predict_with_model(model: Any, frame: np.ndarray, config: Config) -> Any:
    """モデルを使用して予測を実行する関数

    Args:
        model: ロードされたモデル
        frame: 入力フレーム
        config: 設定オブジェクト

    Returns:
        予測結果
    """
    # パラメータを設定
    params = {
        "device": config.device,
        "conf": config.confidence,
        "iou": config.iou,
    }

    # ultralyticsモデルの場合、追加パラメータ
    if USING_ULTRALYTICS:
        params.update(
            {
                "retina_masks": config.retina_masks,
                "imgsz": config.imgsz,
            }
        )

    # モデル実行
    return model(frame, **params)


def process_results(results: Any, frame: np.ndarray, config: Config) -> np.ndarray:
    """予測結果を処理してアノテーションされたフレームを取得する関数

    Args:
        results: モデルの予測結果
        frame: 元のフレーム画像
        config: 設定オブジェクト

    Returns:
        アノテーションされたフレーム
    """
    output_image = frame.copy()

    # 様々な形式のマスク結果に対応
    try:
        masks = None

        # ultralytics形式の場合
        if hasattr(results, "__getitem__") and len(results) > 0:
            if hasattr(results[0], "masks") and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()

        # 他の形式の場合
        elif hasattr(results, "masks") and results.masks is not None:
            if hasattr(results.masks, "data"):
                masks = results.masks.data.cpu().numpy()
            else:
                masks = results.masks

        # マスクが取得できた場合、色付け処理
        if masks is not None and len(masks) > 0:
            num_masks = len(masks)

            # 色の設定
            if config.use_random_colors:
                # ランダムな色を生成（各セグメントに固有の色を割り当てる）
                colors = np.random.randint(50, 255, size=(num_masks, 3), dtype=np.uint8)
            else:
                # 固定色のリスト（色相環に基づいて均等に分布）
                base_colors = []
                for i in range(num_masks):
                    hue = int(180 * i / max(1, num_masks - 1))
                    color = cv2.cvtColor(
                        np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                    )[0][0]
                    base_colors.append(color)
                colors = np.array(base_colors, dtype=np.uint8)

            # 各マスクを処理
            for i, mask in enumerate(masks):
                # マスクの次元に応じた処理
                if isinstance(mask, np.ndarray) and mask.ndim > 1:
                    mask_2d = mask[0] if mask.ndim > 2 else mask
                else:
                    continue

                color = colors[i].tolist()

                # マスクのオーバーレイ
                if config.overlay_mask:
                    mask_image = np.zeros_like(frame, dtype=np.uint8)
                    mask_image[mask_2d > 0.5] = color
                    output_image = cv2.addWeighted(
                        output_image, 1.0, mask_image, config.mask_alpha, 0
                    )

                # マスクの輪郭を描画
                if config.draw_contours:
                    contour_mask = mask_2d.astype(np.uint8)
                    contours, _ = cv2.findContours(
                        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(output_image, contours, -1, color, 2)

            # セグメント数を画像に表示
            if config.show_labels:
                cv2.putText(
                    output_image,
                    f"Segments: {num_masks}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            return output_image

    except Exception as e:
        # エラーが発生した場合、デフォルトの描画方法を試す
        try:
            if hasattr(results, "plot"):
                return results.plot()
            elif hasattr(results, "__getitem__") and hasattr(results[0], "plot"):
                return results[0].plot()
        except:
            pass

    # 何も処理できなかった場合は元のフレームを返す
    return frame


def process_video(input_path: Union[str, int], config: Optional[Config] = None) -> str:
    """動画、画像シーケンス、またはカメラからの入力を処理してセグメンテーションを行う関数

    Args:
        input_path: 処理する入力パス（動画ファイル、画像フォルダ、またはカメラID）
        config: 設定オブジェクト（Noneの場合はデフォルト設定を使用）

    Returns:
        出力ディレクトリのパス
    """
    # 設定の初期化
    if config is None:
        config = Config()

    # 入力パスを設定
    if config.input_type == "camera" and isinstance(input_path, int):
        config.camera_id = input_path
    else:
        config.input_path = input_path

        # 入力ファイル名から適切な出力ファイル名を生成（デフォルト設定の場合のみ）
        if (
            config.output_filename == "segmented_output.mp4"
            and config.input_type == "video"
        ):
            input_filename = os.path.basename(input_path)
            input_name = os.path.splitext(input_filename)[0]
            config.output_filename = f"{input_name}_segmented.mp4"

    # まずトップレベルの出力ディレクトリを作成
    top_output_dir = "output"
    os.makedirs(top_output_dir, exist_ok=True)

    # 次に実行時の出力ディレクトリを作成
    os.makedirs(config.output_dir, exist_ok=True)

    # 出力先を表示
    logger.info(f"出力ディレクトリ: {os.path.abspath(config.output_dir)}")

    if config.save_frames and config.input_type != "camera":
        video_out_path = os.path.join(config.output_dir, config.output_filename)
        logger.info(f"出力動画ファイル: {os.path.basename(video_out_path)}")

    # モデルをロード
    model = load_model(config)

    # 入力ソースを初期化
    source = initialize_input_source(config)
    width, height = source["width"], source["height"]
    fps = source["fps"]

    # 入力サイズを設定
    if config.auto_size and config.imgsz is None:
        # アスペクト比を維持しながら調整（最大1280px）
        max_dim = 1280
        if width > height:
            config.imgsz = min(width, max_dim)
        else:
            config.imgsz = min(height, max_dim)
    elif config.imgsz is None:
        config.imgsz = 640

    # 表示サイズの計算
    display_width = int(width * config.display_scale)
    display_height = int(height * config.display_scale)

    # 出力動画の設定
    out = None
    if config.save_frames and config.input_type != "camera":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    # 処理メイン部分
    frame_count = 0
    try:
        while True:
            # フレームを取得
            ret, frame = get_next_frame(source, frame_count)
            if not ret:
                break

            frame_count += 1

            # モデルで予測を実行
            results = predict_with_model(model, frame, config)
            annotated_frame = process_results(results, frame, config)

            # フレームを保存
            if config.save_frames and out is not None:
                out.write(annotated_frame)
                # 定期的に画像も保存
                if frame_count % config.frame_save_interval == 0:
                    frame_path = os.path.join(
                        config.output_dir, f"frame_{frame_count:06d}.jpg"
                    )
                    cv2.imwrite(frame_path, annotated_frame)

            # フレームを表示
            if config.show_frames:
                cv2.imshow(
                    "FastSAM",
                    cv2.resize(annotated_frame, (display_width, display_height)),
                )

                # キー入力の処理（qで終了、カメラモードでsを押すとスナップショット保存）
                wait_time = 1 if config.input_type == "camera" else 30
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and config.input_type == "camera":
                    snapshot_path = os.path.join(
                        config.output_dir,
                        f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    )
                    cv2.imwrite(snapshot_path, annotated_frame)

    except KeyboardInterrupt:
        pass
    finally:
        # リソースを解放
        if source["cap"] is not None:
            source["cap"].release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    return config.output_dir


def parse_arguments():
    """コマンドライン引数を解析する関数

    Returns:
        解析された引数オブジェクト
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FastSAMを使用して動画/画像/カメラのセグメンテーションを行います",
    )

    # 入力ソースの指定
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="入力動画ファイルのパス")
    input_group.add_argument(
        "--images", type=str, help="入力画像シーケンスのフォルダパス"
    )
    input_group.add_argument(
        "--camera", type=int, default=None, help="カメラデバイスID（通常は0）"
    )

    # 入力設定
    parser.add_argument(
        "--image-pattern", type=str, default="*.jpg", help="画像シーケンスのパターン"
    )
    parser.add_argument(
        "--camera-width", type=int, default=1280, help="カメラキャプチャ幅"
    )
    parser.add_argument(
        "--camera-height", type=int, default=720, help="カメラキャプチャ高さ"
    )
    parser.add_argument(
        "--camera-fps", type=float, default=30.0, help="カメラのFPS設定"
    )

    # モデル設定
    parser.add_argument(
        "--model",
        type=str,
        default="FastSAM-x",
        help="モデル名 (FastSAM-s または FastSAM-x)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="モデルファイルへのパス"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="デバイス (cuda または cpu)"
    )
    parser.add_argument("--force-cpu", action="store_true", help="強制的にCPUを使用")

    # セグメンテーション設定
    parser.add_argument("--conf", type=float, default=0.7, help="信頼度しきい値")
    parser.add_argument("--iou", type=float, default=0.7, help="IOUしきい値")
    parser.add_argument("--imgsz", type=int, default=None, help="入力画像サイズ")
    parser.add_argument(
        "--no-auto-size",
        action="store_false",
        dest="auto_size",
        help="動画サイズに自動調整しない",
    )
    parser.add_argument(
        "--no-retina",
        action="store_false",
        dest="retina",
        help="高品質マスクを使用しない",
    )

    # 出力設定
    parser.add_argument("--output", type=str, default=None, help="出力ディレクトリ")
    parser.add_argument(
        "--output-filename",
        type=str,
        default="segmented_output.mp4",
        help="出力ファイル名",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=30, help="画像保存間隔（フレーム数）"
    )
    parser.add_argument("--display-scale", type=float, default=0.5, help="表示スケール")
    parser.add_argument(
        "--no-save", action="store_false", dest="save", help="フレームを保存しない"
    )
    parser.add_argument(
        "--no-show", action="store_false", dest="show", help="フレームを表示しない"
    )

    # 表示設定
    display_settings = parser.add_argument_group("表示設定")
    display_settings.add_argument(
        "--mask-alpha", type=float, default=0.4, help="マスクの透明度 (0.0-1.0)"
    )
    display_settings.add_argument(
        "--no-labels",
        action="store_false",
        dest="show_labels",
        help="セグメント数などのラベル表示を無効化",
    )
    display_settings.add_argument(
        "--fixed-colors",
        action="store_false",
        dest="use_random_colors",
        help="ランダム色の代わりに固定色を使用",
    )
    display_settings.add_argument(
        "--no-overlay",
        action="store_false",
        dest="overlay_mask",
        help="マスクのオーバーレイを無効化",
    )
    display_settings.add_argument(
        "--no-contours",
        action="store_false",
        dest="draw_contours",
        help="マスクの輪郭描画を無効化",
    )

    # その他の設定
    parser.add_argument(
        "--verbose", action="store_true", help="詳細なログ出力を有効にする"
    )

    return parser.parse_args()


def main():
    """メイン実行関数"""
    # 引数の解析
    args = parse_arguments()

    # ログレベルの設定
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 入力タイプと入力パスの設定
    input_type = None
    input_path = None
    camera_id = 0

    if args.video:
        input_type = "video"
        input_path = args.video
    elif args.images:
        input_type = "images"
        input_path = args.images
    elif args.camera is not None:
        input_type = "camera"
        camera_id = args.camera

    # デバイス設定
    device = "cpu" if args.force_cpu else args.device

    # 設定オブジェクト作成
    config = Config(
        input_type=input_type,
        input_path=input_path,
        camera_id=camera_id,
        image_pattern=args.image_pattern,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        model_name=args.model,
        model_path=args.model_path,
        device=device,
        confidence=args.conf,
        iou=args.iou,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        auto_size=args.auto_size,
        save_frames=args.save,
        show_frames=args.show,
        frame_save_interval=args.frame_interval,
        display_scale=args.display_scale,
        output_dir=args.output,
        output_filename=args.output_filename,
        # 表示設定
        mask_alpha=args.mask_alpha,
        show_labels=args.show_labels,
        use_random_colors=args.use_random_colors,
        overlay_mask=args.overlay_mask,
        draw_contours=args.draw_contours,
    )

    # 処理を開始
    process_video(input_path if input_type != "camera" else camera_id, config)


if __name__ == "__main__":
    """
    FastSAM Video Segmentation Tool

    このツールは、動画、画像シーケンス、またはカメラからの入力に対して
    FastSAMモデルを使用したセマンティックセグメンテーションを行います。

    必要なもの:
    - FastSAMモデル (FastSAM-s.pt または FastSAM-x.pt)
      - CASIA-IVA-Labの公式リポジトリからダウンロード: https://github.com/CASIA-IVA-Lab/FastSAM
      - モデルファイルは weights/ ディレクトリに配置してください

    使用方法:
    1. 動画ファイルのセグメンテーション:
       python video_segment.py --video path/to/video.mp4

    2. 画像シーケンスのセグメンテーション:
       python video_segment.py --images path/to/image/folder --image-pattern "*.jpg"

    3. カメラ入力のリアルタイムセグメンテーション:
       python video_segment.py --camera 0

    出力:
    - output/[タイムスタンプ]/ ディレクトリに結果が保存されます
    - セグメンテーション結果の動画とキーフレーム画像が出力されます

    カスタマイズ:
    - 表示設定 (--mask-alpha, --fixed-colors, --no-labels など)
    - セグメンテーション設定 (--conf, --iou など)

    詳細な使用方法は以下のコマンドで確認できます:
       python video_segment.py --help
    """
    # ロゴ表示
    print("=" * 60)
    print(" FastSAM Video Segmentation Tool")
    print("=" * 60)

    main()
