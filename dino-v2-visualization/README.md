# DINOv2 特徴量可視化・検証スクリプト

このスクリプトは、DINOv2モデルから抽出された特徴量を可視化し、検証するための基本的なフレームワークを提供します。画像の特徴量に対してPCA分析を行い、その結果を可視化することができます。

## 機能

*   複数のDINOv2モデルをサポート:
    *   `ViT-s/14`
    *   `ViT-b/14`
    *   `ViT-l/14`
    *   `ViT-g/14`
*   モデルIDと画像フォルダのパスをコマンドライン引数で指定可能
*   画像の特徴量に対してPCA分析を実行
*   PCA分析の結果を画像として可視化

## 必要な環境

requirements.lock参照
必要なパッケージはryeでインストールできます:

```bash
rye sync
```

## 参考
https://github.com/JunukCha/DINOv2_pca_visualization/blob/main/main.py
https://github.com/purnasai/Dino_V2



