# FastSAM 動画セグメンテーションツール

![FastSAM](https://img.shields.io/badge/FastSAM-Segment%20Anything%20Fast-blue)
![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

このツールは、Ultralytics 社の FastSAM モデルを使用して動画、画像シーケンス、またはカメラのリアルタイム映像のセグメンテーションを行うための Python スクリプトです。各フレームに対してセグメンテーションを適用し、結果を動画および定期的なフレーム画像として保存します。

<img src="https://github.com/CASIA-IVA-Lab/FastSAM/raw/main/assets/showcase.png" alt="FastSAM Showcase" width="600"/>

## 📋 機能

- **複数の入力ソースに対応**:
  - 📹 動画ファイルのセグメンテーション
  - 🖼️ 画像シーケンス（連番画像）のセグメンテーション
  - 📷 Web カメラなどからのリアルタイム入力のセグメンテーション
- 🖥️ GPU の自動検出とフォールバック機能
- 🔄 入力動画サイズに合わせた最適な処理サイズの自動調整
- 💾 セグメンテーション結果の動画保存
- 📊 定期的なフレームの JPEG 画像として保存
- 👁️ リアルタイムでのプレビュー表示
- 📸 カメラモードでのスナップショット保存機能

## 🔧 必要条件

- Python 3.6 以上
- 以下の Python パッケージ:
  - ultralytics または FastSAM リポジトリ
  - opencv-python
  - numpy
  - torch

## ⚙️ インストール方法

### 方法 1: ultralytics パッケージを使用する場合（推奨）

1. 必要なパッケージをインストールします:

```bash
pip install ultralytics opencv-python numpy torch
```

2. FastSAM モデルをダウンロードします（初回実行時に自動でダウンロードされます）:

```bash
# FastSAM-s（高速・軽量モデル）
python -c "from ultralytics import FastSAM; FastSAM('FastSAM-s')"

# または FastSAM-x（高精度モデル）
python -c "from ultralytics import FastSAM; FastSAM('FastSAM-x')"
```

### 方法 2: FastSAM リポジトリを直接使用する場合

最新の ultralytics パッケージで FastSAM をインポートできない場合は、オリジナルの FastSAM リポジトリを使用する方法もあります：

1. 必要なパッケージをインストールします:

```bash
pip install opencv-python numpy torch
```

2. FastSAM リポジトリをクローンします:

```bash
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt
```

3. モデルをダウンロードします:

   - [FastSAM-x](https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt) (138MB)
   - [FastSAM-s](https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-s.pt) (23MB)

4. ダウンロードしたモデルをスクリプトと同じディレクトリに配置します。

## 🚀 使用方法

### 動画ファイルの処理

```bash
python video_segment.py --video <動画ファイルのパス>
```

### 画像シーケンスの処理

```bash
python video_segment.py --images <画像フォルダのパス> --image-pattern "*.jpg"
```

### カメラのリアルタイム入力の処理

```bash
python video_segment.py --camera 0  # 0はカメラのデバイスID
```

これらのコマンドにより、セグメンテーション処理が開始されます。デフォルトでは:

- 入力サイズは動画/画像の解像度に自動調整されます（最大 1280px）
- GPU が利用可能な場合は自動的に使用されます
- 結果は「output_YYYYMMDD_HHMMSS」フォルダに保存されます

### オリジナルの FastSAM リポジトリを使用する場合

FastSAM リポジトリを直接使用する場合は、モデルファイルのパスを指定してください：

```bash
python video_segment.py --video <動画ファイルのパス> --model-path path/to/FastSAM.pt
```

## ⚙️ オプション

| オプション                         | 説明                                                 | デフォルト値         |
| ---------------------------------- | ---------------------------------------------------- | -------------------- |
| **入力ソース（いずれか一つ必須）** |                                                      |                      |
| --video                            | 処理する動画ファイルのパス                           | -                    |
| --images                           | 処理する画像シーケンスのフォルダパス                 | -                    |
| --camera                           | カメラデバイス ID                                    | -                    |
| **入力設定**                       |                                                      |                      |
| --image-pattern                    | 画像シーケンスの検索パターン                         | \*.jpg               |
| --camera-width                     | カメラキャプチャの幅                                 | 1280                 |
| --camera-height                    | カメラキャプチャの高さ                               | 720                  |
| --camera-fps                       | カメラの FPS 設定                                    | 30.0                 |
| **モデル設定**                     |                                                      |                      |
| --model                            | 使用するモデル（FastSAM-s または FastSAM-x）         | FastSAM-x            |
| --model-path                       | モデルファイルへの直接パス                           | None                 |
| --device                           | 使用するデバイス（cuda/cpu、自動検出の場合は未指定） | 自動検出             |
| --force-cpu                        | GPU が利用可能でも強制的に CPU を使用                | False                |
| **セグメンテーション設定**         |                                                      |                      |
| --conf                             | 検出の信頼度しきい値                                 | 0.4                  |
| --iou                              | IOU しきい値                                         | 0.7                  |
| --imgsz                            | 入力画像サイズ（未指定の場合は動画サイズに合わせる） | 自動                 |
| --no-auto-size                     | 動画サイズに自動的に合わせない                       | False                |
| --no-retina                        | 高品質マスクを使用しない                             | False                |
| **出力設定**                       |                                                      |                      |
| --output                           | 出力ディレクトリ                                     | 自動生成             |
| --output-filename                  | 出力ファイル名                                       | segmented_output.mp4 |
| --frame-interval                   | 画像保存間隔（フレーム数）                           | 30                   |
| --display-scale                    | 表示スケール（1.0 で原寸大）                         | 0.5                  |
| --no-save                          | フレームを保存しない                                 | False                |
| --no-show                          | フレームを表示しない                                 | False                |

## 💻 使用例

### 動画ファイルを処理

```bash
python video_segment.py --video sample/video.mp4
```

### 画像シーケンスを処理（PNG ファイル）

```bash
python video_segment.py --images sample/frames/ --image-pattern "*.png"
```

### Web カメラからリアルタイム入力を処理（高解像度設定）

```bash
python video_segment.py --camera 0 --camera-width 1920 --camera-height 1080
```

### 軽量モデルでパフォーマンス重視の処理

```bash
python video_segment.py --video sample/video.mp4 --model FastSAM-s --imgsz 320 --no-retina
```

### 精度重視の設定

```bash
python video_segment.py --video sample/video.mp4 --conf 0.5 --iou 0.6
```

### オリジナルリポジトリのモデルを使用

```bash
python video_segment.py --video sample/video.mp4 --model-path FastSAM-x.pt
```

## 🎮 操作方法

- 表示ウィンドウで`q`キーを押すと処理を終了します
- カメラモードで`s`キーを押すと現在のフレームのスナップショットを保存します

## 🚀 パフォーマンス向上のヒント

- **処理速度を優先する場合**:

  - `--model FastSAM-s`を使用する
  - 処理サイズを小さくする（`--imgsz 320`など）
  - `--no-retina`オプションを使用して高品質マスクを無効化

- **精度を優先する場合**:
  - `--model FastSAM-x`を使用する（デフォルト）
  - 入力サイズを大きくする（または自動設定を利用）
  - 信頼度しきい値を調整する（`--conf 0.6`など）

## ❓ トラブルシューティング

### ImportError: cannot import name 'FastSAM' from 'ultralytics'

最新バージョンの`ultralytics`パッケージでは、FastSAM のインポート方法が変更されている可能性があります。以下の解決方法を試してください：

1. ultralytics パッケージを最新バージョンに更新：

   ```bash
   pip install -U ultralytics
   ```

2. それでも解決しない場合は、オリジナルの FastSAM リポジトリを使用：
   ```bash
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   cd FastSAM
   pip install -r requirements.txt
   ```
   そして、モデルパスを直接指定して実行：
   ```bash
   python video_segment.py --video sample/video.mp4 --model-path FastSAM-x.pt
   ```

## 📚 出力ファイル

- セグメンテーションされた動画: `<出力ディレクトリ>/segmented_output.mp4`
- サンプルフレーム画像: `<出力ディレクトリ>/frame_XXXXXX.jpg`（30 フレームごと）
- カメラモードでのスナップショット: `<出力ディレクトリ>/snapshot_YYYYMMDD_HHMMSS.jpg`

## ⚠️ 注意点

- GPU メモリが足りない場合は自動的に CPU にフォールバックします
- 大きな解像度の動画や画像では処理に時間がかかる場合があります
- カメラ入力では、カメラのサポートする解像度によって実際の出力サイズが制限される場合があります
- 画像シーケンスは名前順にソートされて処理されます

## 📜 ライセンス

このプロジェクトは MIT ライセンスのもとで提供されています。

## 👏 謝辞

- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) - 高速なセグメンテーションモデル
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO と SAM の実装
