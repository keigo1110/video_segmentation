# MobileSAM と FastSAM による動画セグメンテーション

このリポジトリでは、MobileSAM（Mobile Segment Anything Model）と FastSAM（Fast Segment Anything Model）を使用して動画、画像シーケンス、カメラ入力に対してセグメンテーション処理を行うツールを提供しています。両モデルはオリジナルの SAM（Segment Anything Model）の軽量版であり、それぞれ異なる特性と用途に最適化されています。

## インストール

### 前提条件

- CUDA 対応 GPU を推奨

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

## 使用方法

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

## プロジェクト構造

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
└── .gitignore             # Git除外設定ファイル
```

## 詳細なドキュメント

各モデルの詳細な使用方法、パラメータ設定、オプションについては、それぞれのディレクトリ内の README を参照してください:

- [MobileSAM の詳細ドキュメント](./MobileSAM/README.md)
- [FastSAM の詳細ドキュメント](./FastSAM/README.md)
