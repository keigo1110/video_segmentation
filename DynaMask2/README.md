# DynaMask2: Dynamic Object Masking Tool

DynaMask2は、動画、画像シーケンス、またはカメラ入力から、**実世界で動いているオブジェクトのみ**を検出し、そのマスクを生成するツールです。カメラ自体の動き（例: パン、チルト、ズーム）による背景の見かけ上の動きを考慮し、真に動的な要素（人物、車両など）を選択的に抽出することを目的としています。

## ✨ 主な特徴

*   **動的要素の選択的抽出:** カメラの動きを補正し、背景など静的な要素の動きを除外して、本当に動いているオブジェクトのみをマスクします。
*   **複数技術の組み合わせ:**
    *   **FastSAM:** 高速セマンティックセグメンテーションでオブジェクト候補を効率的に検出します。
    *   **オプティカルフロー:** フレーム間のピクセルレベルの動きを計算し、動きの情報を抽出します。
    *   **COLMAPベースの自己運動分離:** COLMAPから得られるカメラポーズ情報を用いて、カメラ自身の動きによる見かけ上の移動を補正します。これにより、静止背景上での物体の微小な動きも検出しやすくなります。
    *   **領域適応閾値処理:** 手の周辺領域とそれ以外の背景領域で、動き検出の感度（閾値）を自動調整します。手の周りは敏感に、背景は鈍感にすることで、手やツールの細かな動きを捉えつつ背景ノイズを抑制します。
    *   **人間検出 (YOLOv8/MediaPipe):** 人物や手を高精度で検出し、動的要素として優先的に考慮します。
*   **柔軟な設定:** コマンドライン引数や設定ファイルを通じて、動き検出の閾値、セグメント判定基準、人間検出の有効/無効、使用モデル、新機能のパラメータなどを細かく調整可能です。
*   **モジュール化設計:** 機能ごとにコードが分割されており、特定機能の改善や他の手法への差し替えが比較的容易です。
*   **並列処理:** 処理速度向上のため、複数のコア処理（動き検出、人間検出、セグメンテーション）を並行して実行します。

## 🔧 インストール

1.  **リポジトリのクローン:**
    ```bash
    git clone <your-repository-url>
    cd DynaMask2
    ```
2.  **(推奨) 仮想環境の作成と有効化:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **依存関係のインストール:**
    *   **PyTorch:** ご利用の環境（OS, CUDAバージョン）に合わせて、[PyTorch公式サイト](https://pytorch.org/get-started/locally/) の指示に従ってインストールしてください。
        ```bash
        # 例 (CUDA 12.1 の場合):
        # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **その他のライブラリ:** `requirements.txt` を使ってインストールします。
        ```bash
        pip install -r requirements.txt
        ```
4.  **DynaMask2 パッケージのインストール:**
    ```bash
    pip install .
    ```
    これにより、`dynamask` コマンドが利用可能になります。

5.  **`video_segment.py` の準備:**
    *   **重要:** このリポジトリは FastSAM の機能を `dynamask/video_segment.py` に依存しています。元の FastSAM 実装から `video_segment.py` （またはそれに相当するファイル）の内容**全体**を `DynaMask2/dynamask/video_segment.py` にコピーしてください。（ライセンスにご注意ください）

6.  **モデルファイルの準備:**
    *   FastSAMモデル (`FastSAM-x.pt` など) と YOLOモデル (`yolov8n.pt` など) をリポジトリのルートディレクトリ、または `config.py` やコマンドライン引数で指定された場所に配置してください。

## 🚀 使い方

インストール後、`dynamask` コマンドを使用できます。コマンドラインから主要な設定を変更できます。

```bash
# ヘルプ表示 (全オプション確認)
dynamask --help
```

### 基本的な実行

```bash
# 例: 動画ファイルを入力とし、結果を ./output_video に保存
# マスク画像とデバッグ画像も保存する
dynamask --input_type video --input_path /path/to/your/video.mp4 \
         --output_dir ./output_video --save_masks --save_debug_frames

# 例: 画像シーケンスを入力とし、結果を ./output_images に保存
dynamask --input_type images --input_path /path/to/image_folder \
         --image_pattern "*.jpg" --output_dir ./output_images

# 例: カメラ入力 (ID: 0) を使用
dynamask --input_type camera --input_path 0 --output_dir ./output_camera
```

### COLMAP 自己運動分離の利用

事前に COLMAP を実行し、カメラポーズ情報 (`cameras.txt`, `images.txt`) を取得している場合、以下のオプションで自己運動分離機能を有効にできます。

```bash
# 例: 画像シーケンス入力でCOLMAPデータを使用
dynamask --input_type images --input_path /path/to/image_folder \
         --output_dir ./output_colmap \
         --colmap_cameras_path /path/to/colmap/cameras.txt \
         --colmap_images_path /path/to/colmap/images.txt \
         --save_masks --save_debug_frames
```

*   `--colmap_cameras_path`: COLMAP の `cameras.txt` ファイルへのパス。
*   `--colmap_images_path`: COLMAP の `images.txt` ファイルへのパス。
*   `--use_colmap_egomotion`: (オプション) `False` を指定すると、COLMAPファイルが指定されていても自己運動分離を無効化します (デフォルト: `True`)。
*   `--fallback_to_flow_compensation`: (オプション) COLMAP分離が失敗した場合や無効の場合に、従来のオプティカルフロー中央値補正を行うか (デフォルト: `True`)。

### 領域適応閾値の調整

手の検出が有効な場合、手の周辺と背景で動き検出の閾値が自動調整されます。以下のパラメータで調整できます。

```bash
# 例: 手の周辺領域の半径と閾値比率を変更
dynamask --input_type video --input_path /path/to/your/video.mp4 \
         --output_dir ./output_adaptive \
         --hand_region_radius 150 \
         --hand_region_threshold_factor 0.3
```

*   `--hand_region_radius`: 手の位置を中心とした作業領域の半径 (ピクセル単位、デフォルト: 100)。
*   `--hand_region_threshold_factor`: 背景領域の閾値に対する作業領域の閾値の比率 (デフォルト: 0.5)。値が小さいほど手の周りで敏感になります。
*   `--use_region_adaptive_threshold`: (オプション) `False` を指定すると、領域適応閾値を無効化します (デフォルト: `True`)。

### その他の主要オプション

*   `--motion_threshold`: 動き検出の基本閾値スケール (デフォルト: 80.0)。適応的閾値の計算に影響します。
*   `--no-yolo` / `--no-pose-detection`: 特定の人間検出方法を無効化します。
*   `--save_masks`: 動的マスク画像を保存します。
*   `--save_debug_frames`: 動きマスクなどのデバッグ用中間画像を保存します。

その他のオプションについては `--help` を参照してください。

### 出力

*   **動画:** 指定された出力ディレクトリ (`output_dir`) に、動的オブジェクトがハイライトされた動画ファイル (`output_filename`) が生成されます。
*   **マスク画像 (`--save_masks`):** `output_dir/masks` に、各フレームの動的マスク（動的領域=黒, 静的領域=白）が PNG 形式で保存されます。画像シーケンス入力の場合は、元のファイル名に基づいて保存されます。
*   **デバッグ画像 (`--save_debug_frames`):** `output_dir/debug` に、ワーピング結果、動きマスク、人間マスク、手の位置、領域マスクなどの中間結果が保存されます。

## ⚙️ 設定ファイル (config.py)

コマンドライン引数で指定できるオプションは、`dynamask/config.py` 内の `DynaMaskConfig` クラスで定義されたパラメータに対応しています。より詳細な設定やデフォルト値の変更は、このファイルを直接編集するか、独自の Config オブジェクトをプログラムから渡すことで可能です。

## 🏗️ プロジェクト構造 (概要)

*   `dynamask/`: コア機能の Python パッケージ。
    *   `pipeline.py`: メイン処理フロー。
    *   `config.py`: 設定クラス。
    *   `motion.py`: 動き検出 (フロー、COLMAPワーピング、領域適応閾値)。
    *   `detection.py`: 人間検出 (YOLO/MediaPipe)。
    *   `filtering.py`: 動的セグメント判定ロジック。
    *   `io.py`: 入出力、COLMAPデータ読み込み。
    *   `utils.py`: ユーティリティ関数 (ログ設定、描画、座標変換)。
    *   `video_segment.py`: FastSAM 関連 (外部依存)。
*   `run_dynamask_cli.py`: コマンドラインインターフェース。
*   `setup.py`, `requirements.txt`: パッケージングと依存関係。
*   `README.md`, `develop.md`: ドキュメント。

## 🧑‍💻 開発者向け情報

詳細な実装計画や技術的な背景については `develop.md` を参照してください。

## 📜 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。（`setup.py` および必要に応じて `LICENSE` ファイルを作成・修正してください）
ただし、依存している FastSAM (`dynamask/video_segment.py` にコピーするコード) およびそのモデルのライセンスは別途ご確認ください。

## 🙌 貢献

バグ報告、機能提案、プルリクエストを歓迎します。

## 🙏謝辞 (オプション)

(もしあれば) 参考にしたり、データセットを提供したりした研究やプロジェクトへの謝辞を記載します。
