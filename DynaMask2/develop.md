1.  **COLMAPベースの自己運動分離:** COLMAP形式のカメラポーズファイル (`images.txt`, `cameras.txt`) が存在する場合に、それを用いて自己運動（カメラ自身の動き）による見かけ上の動きを除去し、実際の物体の動きのみを検出する。
2.  **領域適応閾値処理:** 手の周辺領域とそれ以外の背景領域で、動き検出の感度（閾値）を変える。手の周りは敏感に、背景は鈍感にすることで、手やツールの細かな動きを捉えつつ背景のノイズを抑制する。

---

**修正計画**

**フェーズ1: 設定とデータ構造の準備**

1.  **`config.py` の `DynaMaskConfig` クラスへのパラメータ追加:**
    * **COLMAP関連:**
        * `colmap_images_path: Optional[str] = None`: `images.txt` ファイルへのパス。
        * `colmap_cameras_path: Optional[str] = None`: `cameras.txt` ファイルへのパス。
        * `use_colmap_egomotion: bool = True`: COLMAPファイルが存在する場合に自己運動分離を試みるかどうかのフラグ。デフォルトは`True`とする。
        * `fallback_to_flow_compensation: bool = True`: COLMAP分離が失敗した場合や無効の場合に、既存のオプティカルフロー中央値補正を行うかのフラグ。
    * **領域適応閾値関連:**
        * `use_region_adaptive_threshold: bool = True`: 領域適応閾値処理を有効にするフラグ。要件通り常に適用するため、デフォルト`True`。
        * `hand_region_radius: int = 100`: 手の位置を中心とした作業領域の半径（ピクセル単位）。
        * `hand_region_threshold_factor: float = 0.5`: 背景領域の閾値に対する作業領域の閾値の比率（例: 0.5なら作業領域の閾値は背景の半分）。
    * **既存パラメータの確認:**
        * `camera_compensation`: `fallback_to_flow_compensation` が`True`の場合に使用される。

2.  **COLMAPデータ構造の定義:**
    * `cameras.txt` を読み込んだ結果を保持するデータクラスまたは辞書構造を定義する（例: `camera_id` をキーとし、`width`, `height`, `fx`, `fy`, `cx`, `cy` を含むオブジェクトを値とする）。
    * `images.txt` を読み込んだ結果を保持するデータクラスまたは辞書構造を定義する（例: `image_id` または `image_name` をキーとし、`qvec` (クォータニオン), `tvec` (並進ベクトル), `camera_id` を含むオブジェクトを値とする）。

**フェーズ2: COLMAPデータ読み込み機能の実装**

1.  **実装場所の決定:** COLMAP関連の処理は外部データに依存するため、`io.py` に実装するのが適切でしょう。
2.  **`io.py` に以下の関数を追加:**
    * `load_colmap_cameras(cameras_txt_path: str) -> Dict[int, Any]`:
        * `cameras.txt` を開き、コメント行 (`#`) をスキップして各行をパースする。
        * カメラID、モデル、幅、高さ、カメラパラメータ（焦点距離、主点座標など）を抽出する。
        * カメラIDをキー、カメラ情報を格納したオブジェクト（または辞書）を値とする辞書を構築して返す。
        * ファイルが存在しない、またはフォーマットが不正な場合のエラーハンドリングを追加する。
    * `load_colmap_images(images_txt_path: str) -> Dict[int, Any]`:
        * `images.txt` を開き、コメント行 (`#`) をスキップして各行をパースする。
        * 各画像について、画像ID、`qvec`（クォータニオン）、`tvec`（並進ベクトル）、`camera_id`、画像ファイル名 (`name`) を抽出する。注意: `images.txt` は2行で1画像の情報を持つ形式。
        * 画像IDをキー、画像情報（ポーズ、対応カメラID、ファイル名など）を格納したオブジェクト（または辞書）を値とする辞書を構築して返す。ファイル名 (`name`) をキーとする辞書も別途作成すると後で便利かもしれない。
        * ファイルが存在しない、またはフォーマットが不正な場合のエラーハンドリングを追加する。
    * **ヘルパー関数 (オプション):** クォータニオンから回転行列への変換関数 (`qvec_to_rotmat`) を `utils.py` などに追加すると良い。

**フェーズ3: 自己運動分離ロジックの実装 (`motion.py`)**

1.  **`detect_motion_optical_flow` 関数のシグネチャ変更:**
    * COLMAPデータと手の位置を受け取れるように引数を追加・変更する。
        * `config: DynaMaskConfig`
        * `curr_frame: np.ndarray`
        * `prev_frame: np.ndarray`
        * `curr_frame_info: Optional[Dict] = None`: 現在フレームのCOLMAP画像情報 (ポーズ、ファイル名など)。
        * `prev_frame_info: Optional[Dict] = None`: 前フレームのCOLMAP画像情報。
        * `camera_params: Optional[Dict] = None`: 対応するカメラの内部パラメータ。
        * `prev_hand_positions: Optional[List[Tuple[int, int]]] = None`: **前フレームで検出された**手の位置リスト。

2.  **関数の処理フロー変更:**
    * **Step 1: 自己運動分離方法の決定:**
        * `config.use_colmap_egomotion` が `True` かつ `curr_frame_info`, `prev_frame_info`, `camera_params` がすべて提供されているかチェック。
        * 提供されている場合、`colmap_separation = True` とする。
        * 提供されていない、またはフラグが `False` の場合、`colmap_separation = False` とする。
    * **Step 2: COLMAPベースの自己運動分離 (if `colmap_separation == True`):**
        * 前フレームのポーズ $P_p = (R_p, T_p)$ と現在のフレームのポーズ $P_c = (R_c, T_c)$ を取得。回転は行列形式に変換。
        * カメラ内部パラメータ行列 $K$ を `camera_params` から構築。
        * **ワーピングマップの計算:**
            * 現在のフレーム $I_c$ の各ピクセル $(u_c, v_c)$ について処理。
            * **逆投影:** $(u_c, v_c)$ を正規化画像座標 $(x'_c, y'_c) = K^{-1} [u_c, v_c, 1]^T$ に変換。
            * **深度の仮定:** ここで深度 $d_c$ が必要。深度カメラがない場合は、シーンの支配的な平面を仮定するか、固定の深度を使う。固定深度 $D$ を使う場合、3D点 $X_c = D \cdot [x'_c, y'_c, 1]^T$ を現在のカメラ座標系で得る。
            * **座標系変換:** $X_c$ を前フレームのカメラ座標系 $X_p$ に変換: $X_p = R_p R_c^T (X_c - T_c) + T_p$。（注意：$T$ が world->camera か camera->world かで式が変わる。COLMAPは通常 world->camera）。
            * **再投影:** $X_p$ を前フレームの画像座標 $(u_p, v_p)$ に投影: $[u_p', v_p', w_p']^T = K X_p$。$u_p = u_p'/w_p'$, $v_p = v_p'/w_p'$。
            * 現在のピクセル $(u_c, v_c)$ に対応する前フレームの座標 $(u_p, v_p)$ を計算し、これを `cv2.remap` のマップ (`map1`, `map2`) として全ピクセル分構築する。
        * **ワーピング実行:** `warped_prev_gray = cv2.remap(prev_gray, map1, map2, cv2.INTER_LINEAR)`。
        * **差分計算:** `diff_frame = cv2.absdiff(curr_gray, warped_prev_gray)`。
        * **動きマスク生成:** 差分画像 `diff_frame` を閾値処理して初期動きマスク `initial_motion_mask` を生成する。**この閾値処理に領域適応を適用する。** (詳細は後述のStep 4)。
        * 処理中にエラーが発生した場合（例: ポーズがない）、`colmap_separation = False` に設定し、ログを出力してオプティカルフローベースの処理にフォールバックする。
    * **Step 3: オプティカルフローベースの処理 (if `colmap_separation == False`):**
        * 現状の `cv2.calcOpticalFlowFarneback` を実行してフロー `flow` を計算。
        * **フロー補正:** `config.fallback_to_flow_compensation` が `True` なら中央値補正を実行。
        * マグニチュード `mag` を計算。
        * **動きマスク生成:** マグニチュード `mag` を閾値処理して初期動きマスク `initial_motion_mask` を生成する。**この閾値処理に領域適応を適用する。** (詳細は後述のStep 4)。
    * **Step 4: 領域適応閾値処理の適用 (動きマスク生成部分):**
        * このステップは Step 2 (COLMAP差分) または Step 3 (フローマグニチュード) の閾値処理部分で実行される。
        * 入力は閾値処理対象の強度マップ (`diff_frame` または `mag`)。
        * `config.use_region_adaptive_threshold` が `True` かチェック (要件通りなら常にTrue)。
        * **作業領域マスク生成:** `prev_hand_positions` (前フレームの手の位置) を使用。
            * `work_region_mask = np.zeros(curr_gray.shape, dtype=np.uint8)` を初期化。
            * 各 `(hx, hy)` in `prev_hand_positions` について、`cv2.circle(work_region_mask, (hx, hy), config.hand_region_radius, 255, -1)` を実行。
            * `background_mask = cv2.bitwise_not(work_region_mask)` を計算。
        * **閾値の計算:**
            * **背景閾値 $T_{bg}$:**
                * フローベースの場合: 現状の適応的閾値 `adaptive_threshold = mean_mag + std_mag * adaptive_scale` を計算。これを $T_{bg}$ とする。
                * COLMAP差分ベースの場合: 差分画像の平均値 `mean_diff` と標準偏差 `std_diff` を計算し、`T_{bg} = mean_diff + std_diff * adaptive_scale` のように適応的閾値を計算する。`adaptive_scale` は `config.motion_threshold` から調整する。
            * **作業領域閾値 $T_{work}$:** $T_{work} = T_{bg} \times \text{config.hand\_region\_threshold\_factor}$ を計算。
        * **領域別閾値処理:**
            * `initial_motion_mask = np.zeros_like(curr_gray, dtype=np.uint8)`
            * `intensity_map = diff_frame if colmap_separation else mag` # 閾値処理対象
            * `initial_motion_mask[work_region_mask > 0 & (intensity_map > T_{work})] = 255`
            * `initial_motion_mask[background_mask > 0 & (intensity_map > T_{bg})] = 255`
    * **Step 5: 後処理:**
        * 生成された `initial_motion_mask` に対して、現状の形態素演算（Open, Close）と連結成分除去を適用して最終的な `motion_mask` を得る。
    * **Step 6: 戻り値:** `motion_mask` を返す。

**フェーズ4: パイプラインの修正 (`pipeline.py`)**

1.  **`run_dynamask` 関数の初期化部分:**
    * `config` からCOLMAPファイルのパスを取得。
    * `io.load_colmap_cameras` と `io.load_colmap_images` を呼び出してカメラパラメータと画像ポーズ情報をロードし、変数（例: `colmap_cameras`, `colmap_images_by_id`, `colmap_images_by_name`) に格納。ファイルがない場合やエラーの場合は `None` や空の辞書を格納し、ログを出力。
2.  **フレーム処理ループ内の変更:**
    * **変数準備:**
        * `prev_hand_positions: List[Tuple[int, int]] = []` ループ外で初期化。
        * `prev_frame_info: Optional[Dict] = None` ループ外で初期化。
    * **フレーム情報取得:** 現在のフレーム `frame_idx` または `source_info` からファイル名を取得。`colmap_images_by_name` 辞書を検索して、現在のフレームに対応するCOLMAP情報 `current_frame_info` を取得。
    * **`motion.detect_motion_optical_flow` の呼び出し:**
        * `executor.submit` で呼び出す際の引数を更新。
        * `curr_frame`, `prev_frame`, `current_frame_info`, `prev_frame_info`, `colmap_cameras.get(current_frame_info['camera_id']) if current_frame_info else None`, `prev_hand_positions` を渡す。
    * **状態更新:** ループの最後に `prev_hand_positions = hand_positions` （現在フレームで検出された手の位置）と `prev_frame_info = current_frame_info` を実行し、次のイテレーションに備える。

**フェーズ5: テストとパラメータ調整**

1.  **テストケース作成:**
    * COLMAPファイルが存在するデータセットと存在しないデータセットを用意。
    * 手が映っているシーンと映っていないシーンを用意。
2.  **動作確認:**
    * COLMAPファイルがある場合に、ワーピングと差分による動き検出が機能しているかデバッグ出力で確認。
    * 手の有無や位置によって、動きマスクの感度が変化するか（手の周りでより敏感になるか）確認。
    * COLMAPがない場合に、従来のフローベース＋領域適応閾値処理が機能しているか確認。
3.  **パラメータ調整:**
    * `config.hand_region_radius` と `config.hand_region_threshold_factor` を調整して、手の動きの検出精度と背景ノイズ抑制のバランスをとる。
    * COLMAPベースの自己運動分離で使用する深度仮定や閾値を調整する。
    * `config.motion_threshold` （背景の閾値基準）を再調整する必要があるかもしれない。