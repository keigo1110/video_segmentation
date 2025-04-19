import os
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DynaMaskConfig

logger = logging.getLogger(__name__)

def setup_output_directories(config: 'DynaMaskConfig') -> bool:
    """設定に基づいて出力ディレクトリを作成します。"""
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"メイン出力ディレクトリを作成しました: {config.output_dir}")
        if config.save_debug_frames and config.debug_dir:
            os.makedirs(config.debug_dir, exist_ok=True)
            logger.info(f"デバッグディレクトリを作成しました: {config.debug_dir}")
        if config.save_masks and config.masks_dir:
            os.makedirs(config.masks_dir, exist_ok=True)
            logger.info(f"マスクディレクトリを作成しました: {config.masks_dir}")
        return True
    except OSError as e:
        logger.error(f"出力ディレクトリの作成に失敗しました: {e}")
        return False
    except TypeError as e:
        logger.error(f"出力ディレクトリパスが無効です (Noneなど): {e}")
        return False


def initialize_video_writer(config: 'DynaMaskConfig', width: int, height: int, fps: float) -> Optional[cv2.VideoWriter]:
    """出力動画用の VideoWriter を初期化します。"""
    if not config.output_dir:
        logger.error("出力ディレクトリが設定されていません。VideoWriterを初期化できません。")
        return None

    output_path = os.path.join(config.output_dir, config.output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # H.264などの他のコーデックも検討可能
    try:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
             logger.error(f"VideoWriter を開けませんでした: {output_path}")
             return None
        logger.info(f"出力動画ファイルを開きました: {output_path} ({width}x{height} @ {fps:.2f}fps)")
        return writer
    except Exception as e:
        logger.error(f"VideoWriter の初期化中にエラーが発生しました: {e}", exc_info=True)
        return None

def write_video_frame(writer: Optional[cv2.VideoWriter], frame: np.ndarray) -> bool:
    """フレームを動画ファイルに書き込みます。"""
    if writer is None:
        # logger.warning("VideoWriterが初期化されていないため、フレームを書き込めません。")
        return False # Writerがない場合は単に失敗扱い

    if not writer.isOpened():
         logger.warning("VideoWriterが閉じられているため、フレームを書き込めません。")
         return False

    try:
        writer.write(frame)
        return True
    except Exception as e:
        logger.error(f"動画フレームの書き込み中にエラーが発生しました: {e}", exc_info=True)
        return False


def _generate_output_filename(frame_idx: int, input_type: str, source_info: Optional[Dict[str, Any]] = None) -> str:
    """保存するフレーム/マスクのファイル名を生成します。"""
    frame_name = f"{frame_idx:06d}.png" # デフォルトは連番
    if input_type == "images" and source_info and "file_list" in source_info:
        try:
            # source_info['frame_count'] が 0 の場合のケアが必要
            frame_count = source_info.get("frame_count", 0)
            file_list = source_info["file_list"]
            if frame_count > 0 and len(file_list) > 0:
                 current_frame_num = frame_idx % frame_count
                 if current_frame_num < len(file_list):
                    orig_filename = os.path.basename(file_list[current_frame_num])
                    # 拡張子を .png に統一するかどうか検討 (ここでは元のファイル名基準)
                    base, _ = os.path.splitext(orig_filename)
                    frame_name = f"{base}.png" # 保存形式はPNGに統一
            elif len(file_list) > frame_idx: # フレームカウントがない場合、インデックスでアクセス試行
                 orig_filename = os.path.basename(file_list[frame_idx])
                 base, _ = os.path.splitext(orig_filename)
                 frame_name = f"{base}.png"

        except IndexError:
             logger.warning(f"フレーム {frame_idx} の元のファイル名が見つかりません。連番を使用します。")
        except Exception as e:
            logger.error(f"元のファイル名取得中にエラー: {e}。連番を使用します。")
            # Fallback to default frame_name defined above
    return frame_name


def save_debug_frame(frame: np.ndarray, frame_idx: int, config: 'DynaMaskConfig', prefix: str) -> bool:
    """デバッグ用フレームを指定されたプレフィックスで保存します。"""
    if not config.save_debug_frames or not config.debug_dir:
        return False
    if frame is None or frame.size == 0:
        logger.warning(f"デバッグフレーム {prefix}_{frame_idx:06d} は空です。保存をスキップします。")
        return False

    filename = f"{prefix}_{frame_idx:06d}.png"
    filepath = os.path.join(config.debug_dir, filename)
    try:
        # ディレクトリが存在するか再確認 (makedirsが失敗した場合など)
        if not os.path.exists(config.debug_dir):
             logger.warning(f"デバッグディレクトリ {config.debug_dir} が存在しません。フレームを保存できません。")
             # 再度作成を試みるか、エラーにするか
             try:
                 os.makedirs(config.debug_dir, exist_ok=True)
             except OSError:
                  logger.error(f"デバッグディレクトリ {config.debug_dir} の再作成に失敗しました。")
                  return False

        cv2.imwrite(filepath, frame)
        # logger.debug(f"デバッグフレームを保存しました: {filepath}") # Verbose
        return True
    except Exception as e:
        logger.error(f"デバッグフレームの保存中にエラーが発生しました ({filepath}): {e}", exc_info=True)
        return False

def save_mask_image(mask: np.ndarray, frame_idx: int, config: 'DynaMaskConfig', source_info: Optional[Dict[str, Any]] = None) -> bool:
    """生成されたマスク画像 (動的領域が黒、静的領域が白) を保存します。"""
    if not config.save_masks or not config.masks_dir:
        return False
    if mask is None or mask.size == 0:
        logger.warning(f"マスク画像 {frame_idx} は空です。保存をスキップします。")
        return False
    # Ensure mask is uint8 grayscale
    if mask.dtype != np.uint8:
        logger.warning(f"マスク画像のデータ型が uint8 ではありません ({mask.dtype})。変換を試みます。")
        try:
            mask = mask.astype(np.uint8)
        except Exception as e:
            logger.error(f"マスク画像を uint8 に変換できませんでした: {e}")
            return False
    if len(mask.shape) != 2:
         logger.warning(f"マスク画像がグレースケールではありません (shape: {mask.shape})。保存できません。")
         return False


    filename = _generate_output_filename(frame_idx, config.input_type, source_info)
    filepath = os.path.join(config.masks_dir, filename)
    try:
         # ディレクトリが存在するか再確認
        if not os.path.exists(config.masks_dir):
             logger.warning(f"マスクディレクトリ {config.masks_dir} が存在しません。マスクを保存できません。")
             try:
                 os.makedirs(config.masks_dir, exist_ok=True)
             except OSError:
                  logger.error(f"マスクディレクトリ {config.masks_dir} の再作成に失敗しました。")
                  return False

        cv2.imwrite(filepath, mask)
        # logger.debug(f"マスク画像を保存しました: {filepath}") # Verbose
        return True
    except Exception as e:
        logger.error(f"マスク画像の保存中にエラーが発生しました ({filepath}): {e}", exc_info=True)
        return False

def release_video_writer(writer: Optional[cv2.VideoWriter]):
    """VideoWriter リソースを解放します。"""
    if writer is not None and writer.isOpened():
        try:
            writer.release()
            logger.info("VideoWriter を解放しました。")
        except Exception as e:
            logger.error(f"VideoWriter の解放中にエラーが発生しました: {e}", exc_info=True)

# 注意: video_segment に依存する入力ソースの解放 (cap.release()) は
# pipeline.py または utils.py で別途扱う必要があります。

# ===== COLMAP データ読み込み関数 =====

def load_colmap_cameras(cameras_txt_path: str) -> Dict[int, Dict[str, Any]]:
    """COLMAP の cameras.txt を読み込み、カメラ情報を辞書として返します。

    Args:
        cameras_txt_path: cameras.txt ファイルへのパス。

    Returns:
        キーがカメラID、値がカメラパラメータを含む辞書の辞書。
        例: {1: {'model': 'PINHOLE', 'width': 1920, 'height': 1080, 'params': [fx, fy, cx, cy]}}
    """
    cameras = {}
    try:
        with open(cameras_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    logger.warning(f"cameras.txt の不正な行をスキップ: {line}")
                    continue

                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
        logger.info(f"{len(cameras)} 個のカメラ情報を {cameras_txt_path} から読み込みました。")
    except FileNotFoundError:
        logger.warning(f"カメラファイルが見つかりません: {cameras_txt_path}")
    except ValueError as e:
        logger.error(f"cameras.txt のパース中にエラーが発生しました: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"cameras.txt の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True)
    return cameras

def load_colmap_images(images_txt_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    """COLMAP の images.txt を読み込み、画像情報とファイル名->IDマップを返します。

    Args:
        images_txt_path: images.txt ファイルへのパス。

    Returns:
        タプル:
            - キーが画像ID、値が画像情報（ポーズ、カメラIDなど）を含む辞書の辞書。
            - キーが画像ファイル名、値が画像IDの辞書。
        例: ({1: {'qvec': [w,x,y,z], 'tvec': [x,y,z], 'camera_id': 1, 'name': 'frame001.png', ...}},
             {'frame001.png': 1, ...})
    """
    images_by_id = {}
    images_by_name = {}
    try:
        with open(images_txt_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 9:
                logger.warning(f"images.txt の不正な情報行をスキップ: {line}")
                # 次の POINTS2D 行もスキップする
                if i < len(lines) and not lines[i].strip().startswith('#'):
                    i += 1
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            name = parts[9]

            # qvec と tvec を numpy 配列として格納
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])

            image_data = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': name,
                # points2D は次の行にあるが、ここでは読み込まない
            }
            images_by_id[image_id] = image_data
            images_by_name[name] = image_id

            # 次の行は POINTS2D_XYS なのでスキップする
            if i < len(lines) and not lines[i].strip().startswith('#'):
                i += 1
            else:
                logger.warning(f"画像 {image_id} の POINTS2D 行が見つからないか、コメントです。")

        logger.info(f"{len(images_by_id)} 個の画像情報を {images_txt_path} から読み込みました。")

    except FileNotFoundError:
        logger.warning(f"画像ファイルが見つかりません: {images_txt_path}")
    except ValueError as e:
        logger.error(f"images.txt のパース中にエラーが発生しました: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"images.txt の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True)

    return images_by_id, images_by_name
