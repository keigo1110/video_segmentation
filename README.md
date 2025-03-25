# MobileSAM と FastSAM による動画セグメンテーション

![Python](https://img.shields.io/badge/Python-3.6%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

このリポジトリでは、MobileSAM（Mobile Segment Anything Model）と FastSAM（Fast Segment Anything Model）を使用して動画、画像シーケンス、カメラ入力に対してセグメンテーション処理を行うツールを提供しています。両モデルはオリジナルの SAM（Segment Anything Model）の軽量版であり、それぞれ異なる特性と用途に最適化されています。

<p align="center">
  <img src="https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask.jpg" alt="MobileSAM Example" width="400"/>
  <img src="https://github.com/CASIA-IVA-Lab/FastSAM/raw/main/assets/showcase.png" alt="FastSAM Example" width="400"/>
</p>

## 📋 特徴

- **2 つのモデル**:
  - **MobileSAM**: 高品質なセグメンテーションに最適化され、SAM の精度を維持しながら軽量化
  - **FastSAM**: 非常に高速な推論速度を実現し、リアルタイム処理に最適化
- **複数の入力ソースに対応**:
  - 📹 動画ファイルの処理
  - 🖼️ 画像シーケンス（連番画像）の処理
  - 📷 カメラからのリアルタイム入力
- **柔軟なセグメンテーションオプション**:
  - MobileSAM: ポイントプロンプト、ボックスプロンプト、自動セグメンテーション
  - FastSAM: 高速な全オブジェクトセグメンテーション
- **セグメンテーション結果の視覚化と保存**:
  - 動画出力
  - フレーム画像の定期的な保存
  - リアルタイム表示

## 📊 モデル比較表

| 特性           | MobileSAM                         | FastSAM                                  |
| -------------- | --------------------------------- | ---------------------------------------- |
| モデルサイズ   | 40.7MB                            | 23.7MB (FastSAM-s), 138MB (FastSAM-x)    |
| パラメータ数   | 10.1M                             | 11.8M (FastSAM-s)                        |
| アーキテクチャ | Tiny-ViT ベース                   | YOLOv8 ベース                            |
| 特徴           | 元の SAM と同じパイプラインを維持 | YOLOv8 アーキテクチャを活用した高速処理  |
| 主な利点       | 高品質なセグメンテーション        | 非常に高速な推論速度                     |
| 主なターゲット | モバイル/エッジデバイス           | リアルタイム処理が必要なアプリケーション |

| モデル      | CPU 速度(ms/im) | GPU 速度 | 推論時間 | メモリ使用量 |
| ----------- | --------------- | -------- | -------- | ------------ |
| SAM-b       | 161,440         | 50ms     | 高       | 大           |
| MobileSAM   | 98,543          | 約 40ms  | 中       | 中           |
| FastSAM-s   | 140             | 約 12ms  | 低       | 小           |
| YOLOv8n-seg | 79.5            | 約 3ms   | 最小     | 最小         |

## 🔧 インストール

### 前提条件

- Python 3.6 以上
- CUDA 対応 GPU を推奨（特に MobileSAM は CPU では非常に遅い）

### セットアップ

1. リポジトリをクローン:

```bash
git clone https://github.com/yourusername/segmentation.git
cd segmentation
```

2. 依存パッケージをインストール:

```bash
pip install ultralytics opencv-python numpy torch
```

3. モデルファイルをダウンロード（初回実行時に自動でダウンロードされますが、手動でダウンロードする場合）:

```bash
# MobileSAM
python -c "from ultralytics import SAM; SAM('mobile_sam.pt')"

# FastSAM-s (軽量版)
python -c "from ultralytics import FastSAM; FastSAM('FastSAM-s')"

# FastSAM-x (高精度版)
python -c "from ultralytics import FastSAM; FastSAM('FastSAM-x')"
```

## 🚀 使用方法

### MobileSAM を使用した動画処理

```bash
cd MobileSAM
python video_segment.py --video path/to/video.mp4
```

### FastSAM を使用した動画処理

```bash
cd FastSAM
python video_segment.py --video path/to/video.mp4
```

### その他の入力ソース

画像シーケンスを処理する:

```bash
python video_segment.py --images path/to/image/folder --image-pattern "*.jpg"
```

カメラからのリアルタイム入力を処理する:

```bash
python video_segment.py --camera 0  # 0はカメラのデバイスID
```

## 📖 ユースケース別の推奨モデル

| ユースケース                           | 推奨モデル | 理由                                         |
| -------------------------------------- | ---------- | -------------------------------------------- |
| 高品質なセグメンテーションが必要な場合 | MobileSAM  | 元の SAM に近い精度を維持                    |
| リアルタイム処理が必要な場合           | FastSAM    | CPU でも高速に動作                           |
| モバイルアプリケーション               | MobileSAM  | モバイルに最適化されたアーキテクチャ         |
| バッチ処理の多い場合                   | FastSAM    | 多数の画像を高速に処理可能                   |
| インタラクティブな編集                 | MobileSAM  | 高精度なプロンプトベースのセグメンテーション |
| 組み込みシステム                       | FastSAM-s  | 軽量で高速な処理                             |

## 📁 プロジェクト構造

```
segmentation/
├── MobileSAM/
│   ├── mobile_sam.pt      # MobileSAMモデルファイル
│   ├── video_segment.py   # MobileSAM用の動画セグメンテーションスクリプト
│   ├── README.md          # MobileSAMのドキュメント
│   └── output/            # 出力ファイル用ディレクトリ
│
├── FastSAM/
│   ├── FastSAM-s.pt       # FastSAM軽量モデル
│   ├── FastSAM-x.pt       # FastSAM高精度モデル
│   ├── video_segment.py   # FastSAM用の動画セグメンテーションスクリプト
│   ├── README.md          # FastSAMのドキュメント
│   └── output/            # 出力ファイル用ディレクトリ
│
├── README.md              # メインREADME
├── LICENSE                # MITライセンスファイル
└── .gitignore             # Git除外設定ファイル
```

## 📚 詳細なドキュメント

各モデルの詳細な使用方法、パラメータ設定、オプションについては、それぞれのディレクトリ内の README を参照してください:

- [MobileSAM の詳細ドキュメント](./MobileSAM/README.md)
- [FastSAM の詳細ドキュメント](./FastSAM/README.md)

## 📝 注意点

- MobileSAM は CPU での実行が非常に遅いため、可能な限り GPU の使用をお勧めします
- FastSAM は CPU でも比較的高速に動作しますが、最高のパフォーマンスには GPU をお勧めします
- 大きな解像度の入力では処理に時間がかかる場合があります
- FastSAM-s は軽量ですが精度が低下する場合があります

## 📜 ライセンス

このプロジェクトは MIT ライセンスの下で配布されています。詳細は[LICENSE](./LICENSE)ファイルを参照してください。

## 👏 謝辞

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - 効率的なモバイル SAM の実装
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) - 高速なセグメンテーション実装
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO と SAM の実装

## 📚 参考文献

1. MobileSAM: [Faster Segment Anything: Towards Lightweight SAM for Mobile Applications](https://arxiv.org/abs/2306.14289)
2. FastSAM: [Fast Segment Anything](https://arxiv.org/abs/2306.12156)
3. Ultralytics ドキュメント: [MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)
4. Ultralytics ドキュメント: [FastSAM](https://docs.ultralytics.com/models/fast-sam/)
