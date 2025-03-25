# MobileSAM 動画セグメンテーション

MobileSAM（Mobile Segment Anything Model）を使用して、動画や画像、カメラ入力に対してセグメンテーション処理を行うツールです。MobileSAM は、元の SAM モデルの約 5 倍軽量で、7 倍高速な軽量版のセグメンテーションモデルです。

## 特徴

- **軽量・高速処理**: FastSAM と比較して、非常に軽量かつ高速な MobileSAM を使用
- **複数入力対応**: 動画ファイル、画像シーケンス、カメラ入力に対応
- **柔軟なセグメンテーション**: ポイントプロンプト、ボックスプロンプト、あるいは画像内のすべてのオブジェクトをセグメント
- **カスタマイズ可能な表示**: マスクの透明度、色、輪郭表示などを調整可能

## インストール

必要なライブラリをインストールします:

```bash
pip install ultralytics opencv-python
```

MobileSAM モデルは最初の実行時に自動的にダウンロードされますが、手動でダウンロードする場合：

```bash
python -c "from ultralytics import SAM; SAM('mobile_sam.pt')"
```

## 使用方法

### 基本的な使い方

```bash
python video_segment.py --video <動画ファイルパス>
```

### その他の入力タイプ

画像シーケンスを処理:

```bash
python video_segment.py --images <画像フォルダパス> --image-pattern "*.jpg"
```

カメラ入力を処理:

```bash
python video_segment.py --camera 0
```

### セグメンテーションオプション

画像内のすべてのオブジェクトをセグメント:

```bash
python video_segment.py --video <動画ファイルパス> --everything
```

特定のポイントでセグメント:

```bash
python video_segment.py --video <動画ファイルパス> --points 100 100 200 300
```

特定のボックスでセグメント:

```bash
python video_segment.py --video <動画ファイルパス> --boxes 100 100 300 400
```

### その他のオプション

```bash
# 出力先の指定
python video_segment.py --video <動画ファイルパス> --output <出力ディレクトリ>

# デバイスの指定
python video_segment.py --video <動画ファイルパス> --device cuda

# フレーム表示のスケール調整
python video_segment.py --video <動画ファイルパス> --display-scale 0.7

# マスクの透明度調整
python video_segment.py --video <動画ファイルパス> --mask-alpha 0.5
```

## MobileSAM と FastSAM の比較

| モデル    | サイズ(MB) | パラメータ数(M) | CPU 速度(ms/画像) |
| --------- | ---------- | --------------- | ----------------- |
| FastSAM-s | 23.7       | 11.8            | 140               |
| MobileSAM | 40.7       | 10.1            | 98543             |

MobileSAM はモデルサイズがコンパクトで高速な推論が可能ですが、CPU での実行は遅いため、可能な限り GPU の使用をお勧めします。

## コマンドラインオプション一覧

```
入力設定:
  --video VIDEO         入力動画のパス
  --images IMAGES       入力画像のディレクトリパス
  --camera CAMERA       カメラデバイスID
  --image-pattern IMAGE_PATTERN
                        画像ファイルのパターン (例: '*.jpg')

モデル設定:
  --model MODEL         MobileSAMモデルのパス
  --device DEVICE       使用するデバイス ('cuda' または 'cpu')

セグメンテーション設定:
  --points POINTS [POINTS ...]
                        セグメントするポイント座標 (x1 y1 x2 y2 ...)
  --boxes BOXES [BOXES ...]
                        セグメントするボックス座標 (x1 y1 x2 y2 ...)
  --labels LABELS [LABELS ...]
                        ポイントのラベル (1=ポジティブ, 0=ネガティブ)
  --everything          すべてのオブジェクトをセグメント
  --imgsz IMGSZ         入力画像サイズ

出力設定:
  --output OUTPUT       出力ディレクトリ
  --output-video OUTPUT_VIDEO
                        出力動画ファイル名
  --no-save             フレームを保存しない
  --no-show             リアルタイムでフレームを表示しない
  --frame-interval FRAME_INTERVAL
                        フレーム保存間隔
  --display-scale DISPLAY_SCALE
                        表示スケール

表示設定:
  --mask-alpha MASK_ALPHA
                        マスクの透明度 (0.0-1.0)
  --no-labels           ラベル表示を無効化
  --fixed-color         固定色を使用 (ランダム色を無効化)
  --no-overlay          マスクオーバーレイを無効化
  --no-contours         輪郭描画を無効化
```

## 参考

- [MobileSAM 論文](https://arxiv.org/abs/2306.14289)
- [Ultralytics MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)
