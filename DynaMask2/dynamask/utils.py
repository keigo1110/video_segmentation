import logging
import cv2
import numpy as np
from typing import Tuple, Optional

def setup_logging(level=logging.INFO, log_file: Optional[str] = None):
    """基本的なロギング設定を行います。"""
    handlers = [logging.StreamHandler()] # デフォルトはコンソール出力
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            handlers.append(file_handler)
        except Exception as e:
            logging.warning(f"ログファイル {log_file} を開けませんでした: {e}")

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    logging.info("ロギングを設定しました。")

def draw_text_with_background(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int], # (x, y) bottom-left corner
    font_face = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    thickness: int = 1,
    line_type = cv2.LINE_AA,
    padding: int = 2
):
    """背景付きでテキストを描画します。"""
    try:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness
        )

        # 背景を描画 (padding を考慮)
        if bg_color is not None:
            bottom_left = (org[0] - padding, org[1] + padding + baseline)
            top_right = (org[0] + text_width + padding, org[1] - text_height - padding)
            # 座標が画像範囲内に収まるようにクリップ
            h, w = img.shape[:2]
            bl_x = max(0, bottom_left[0])
            bl_y = min(h - 1, bottom_left[1])
            tr_x = min(w - 1, top_right[0])
            tr_y = max(0, top_right[1])

            # 背景が描画可能か確認
            if bl_x < tr_x and tr_y < bl_y:
                 sub_img = img[tr_y:bl_y, bl_x:tr_x]
                 if sub_img.size > 0: # ROIが空でないことを確認
                    white_rect = np.full(sub_img.shape, bg_color, dtype=img.dtype)
                    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0) # 半透明背景
                    img[tr_y:bl_y, bl_x:tr_x] = res

        # テキストを描画
        cv2.putText(
            img,
            text,
            org,
            font_face,
            font_scale,
            text_color,
            thickness,
            line_type
        )
    except Exception as e:
        # エラーが発生しても処理は続行（描画しないだけ）
        logger = logging.getLogger(__name__) # 関数内でのみロガー取得
        logger.warning(f"テキスト描画エラー: {e}", exc_info=False) # トレースバックは抑制

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """クォータニオン (w, x, y, z) を回転行列に変換します。"""
    if qvec is None or len(qvec) != 4:
        raise ValueError("無効なクォータニオン形式です。")
    # 正規化 (COLMAP出力は通常正規化されているはずだが念のため)
    qvec = qvec / np.linalg.norm(qvec)

    w, x, y, z = qvec
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*z*w
    R[0, 2] = 2*x*z + 2*y*w
    R[1, 0] = 2*x*y + 2*z*w
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*x*w
    R[2, 0] = 2*x*z - 2*y*w
    R[2, 1] = 2*y*z + 2*x*w
    R[2, 2] = 1 - 2*x*x - 2*y*y
    return R
