# SAM2 ローカルインタラクティブセグメンテーションデモ

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- ライセンスに応じて変更 -->

MetaのSegment Anything Model 2 (SAM2) をローカルのGPU環境で実行し、Webブラウザからインタラクティブに操作できるデモアプリケーションです。

## 概要 (Overview)

このプロジェクトは、指定した画像に対し、ユーザーがブラウザ上で点をクリック（含める点/除外する点）することで、SAM2によるセグメンテーション（領域分割）を実行し、結果をリアルタイムに確認できる環境を提供します。

オンラインのデモとは異なり、プライベートな画像を外部にアップロードすることなく、手元のマシンでSAM2の機能を試すことができます。研究開発や性能評価、独自アプリケーションへの組み込み検討などに活用できます。

## 主な機能 (Features)

*   **画像アップロード:** 手元の画像ファイル (JPG, PNG) をブラウザからアップロード。
*   **インタラクティブな点指定:**
    *   画像上でクリックした位置をプロンプトとして使用。
    *   「含める点 (Positive)」と「除外する点 (Negative)」を指定可能。
*   **SAM2によるセグメンテーション:** 指定された点に基づき、バックエンドでSAM2が推論を実行。
*   **結果表示:** マスクが適用された画像と予測スコアをブラウザに表示。
*   **GPU利用:** PyTorchを利用し、ローカルのNVIDIA GPUで高速に推論を実行。
*   **Web UI:** FastAPIによるバックエンドと、シンプルなHTML/JavaScriptによるフロントエンド。

## デモ (Screenshot/GIF)

<!-- ここにアプリケーションが動作しているスクリーンショットやGIFアニメーションを挿入すると分かりやすくなります -->
![GUI画面](images/segmentation.png) <!-- ブログ記事の画像を仮置き -->

## 技術スタック (Tech Stack)

*   **モデル:** Meta Segment Anything Model 2 (SAM2) - Hugging Face `transformers` 経由
*   **バックエンド:** FastAPI, Uvicorn
*   **コア処理:** PyTorch, NumPy, OpenCV (cv2), Pillow (PIL)
*   **フロントエンド:** HTML, JavaScript (Vanilla JS), Tailwind CSS (CDN)
*   **環境管理 (例):** Rye (or pip, Conda)

## 前提条件 (Prerequisites)

*   **Python:** 3.9 以上推奨
*   **NVIDIA GPU:** 推論を高速に行うために必須。
*   **CUDA Toolkit & cuDNN:** PyTorchがGPUを利用するために必要。バージョンはPyTorchの要求に合わせる。
*   **Git:** リポジトリのクローンに必要。
*   **(推奨) Rye:** Python環境とパッケージ管理のため (他のツールでも可)。

## セットアップ (Setup)

1.  **リポジトリをクローン:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```
    ( `YOUR_USERNAME/YOUR_REPOSITORY_NAME` は実際のリポジトリ名に置き換えてください)

2.  **依存関係をインストール:**

    *   **Rye を使用する場合:**
        ```bash
        rye sync
        ```
        (PyTorchのCUDAバージョン指定が必要な場合は `pyproject.toml` を事前に確認・編集してください)

    *   **pip を使用する場合:**
        (仮想環境を作成・有効化することを推奨します)
        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Linux/macOS
        # .venv\Scripts\activate  # Windows
        pip install -r requirements.txt
        ```
        (`requirements.txt` が必要です。Ryeを使っている場合は `rye export --output requirements.txt` などで生成できます。)
        **注意:** PyTorchはCUDAバージョンに合わせて適切なものをインストールする必要があります。PyTorch公式サイト ([https://pytorch.org/](https://pytorch.org/)) で適切なインストールコマンドを確認してください。

3.  **モデルのダウンロード:**
    *   初回実行時に、Hugging Face HubからSAM2のモデルファイルが自動的にダウンロード・キャッシュされます。(`facebook/sam2-vit-h` など、`sam_engine.py` 内で指定されているモデル)
    *   インターネット接続が必要です。キャッシュディレクトリは通常 `~/.cache/huggingface/hub/` などになります。

## 実行方法 (Running the Application)

1.  **FastAPIサーバーを起動:**

    *   **Rye を使用する場合:**
        ```bash
        rye run uvicorn main:app --host 0.0.0.0 --port 8000
        ```

    *   **(仮想環境有効化後) uvicorn を直接使用する場合:**
        ```bash
        uvicorn main:app --host 0.0.0.0 --port 8000
        ```

2.  **ブラウザでアクセス:**
    *   Webブラウザを開き、 `http://localhost:8000` または `http://<サーバーのIPアドレス>:8000` にアクセスします。

## 使い方 (How to Use)

1.  **画像を選択:** "画像を選択してください" ボタンをクリックし、セグメンテーションしたいJPGまたはPNGファイルを選びます。
2.  **点をクリック:**
    *   アップロードされた画像が表示されたら、その上でクリックします。
    *   クリックする前に「含める (Positive)」か「除外 (Negative)」のモードを選択します。
    *   クリックした位置に緑（Positive）または赤（Negative）の点が表示されます。
3.  **セグメンテーション実行:** 点を1つ以上追加したら、「セグメンテーション実行」ボタンをクリックします。
4.  **結果確認:** しばらく待つと、右側にマスクが重なった結果画像と予測スコアが表示されます。
5.  **(任意) 点をクリア:** 「点をクリア」ボタンで指定した点をすべて削除できます。点を追加・削除して再度実行することも可能です。

## 注意点 (Notes)

*   **GPUメモリ:** SAM2 (特にViT-Hベース) は比較的大量のGPUメモリを消費します。メモリ不足の場合は、より小さいモデルバリアント（例: ViT-L, ViT-B）を試すか、画像の解像度を下げるなどの対策が必要になる場合があります。（`sam_engine.py` のモデル名を変更）
*   **推論時間:** 初回の画像セット (`set_image`) や、最初の推論には時間がかかることがあります。一度画像特徴量が計算されれば、点の追加・変更に伴う再推論（`predict`）は高速です。
*   **エラーハンドリング:** このデモのエラーハンドリングは基本的なものです。
*   **アップロードディレクトリ:** アップロードされた画像はデフォルトで `backend/uploaded_images` ディレクトリに保存されます。


