import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from torchvision import transforms

STD = 0.2
MEAN = 0.5
DEVICE = "cuda"
TH = 0.65
PATCH_NUM = 40


def draw_image(features, num_images):
    ncols = 2  # 1行あたりの画像数（2列で固定）
    nrows = (num_images + ncols - 1) // ncols
    # サブプロットのサイズを設定
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    # 画像枚数が奇数の場合、最後のサブプロットを非表示
    if num_images % ncols != 0:
        if nrows > 1:
            # 2次元配列の場合
            axes[-1, -1].axis("off")
        else:
            # 1次元配列の場合
            axes[-1].axis("off")

    for i in range(num_images):
        row = i // ncols
        col = i % ncols

        if nrows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.imshow(features(i))  # 画像をプロット
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")  # 軸を非表示に
    plt.tight_layout()
    plt.show()


def fix_pca_sign(pca_features):
    # 各成分の符号を固定
    # PCAは符号が逆転することもあるので、最初の成分が正になるように調整
    for i in range(pca_features.shape[1]):
        if pca_features[0, i] < 0:
            pca_features[:, i] *= -1
    return pca_features


def get_features(dinov2, image_size, img_path):
    resize_crop_image = transforms.Compose(
        [
            transforms.Resize(image_size + 2),
            transforms.CenterCrop(image_size),  # should be multiple of model patch_size
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    _image = Image.open(img_path).convert("RGB")
    transformed_image = resize_crop_image(_image).unsqueeze(0).to(DEVICE)
    # total_images.append(transformed_image)

    features_dict = dinov2.forward_features(transformed_image)

    features = features_dict["x_norm_patchtokens"]
    return features, transformed_image


def save_images(total_images):
    # transformed_image を images_plot 形式に変換
    images_plot = (
        (
            (torch.cat(total_images, dim=0).cpu().numpy() * STD + MEAN)
            * 255  # 正規化を元に戻し、[0, 255] にスケーリング
        )
        .astype(np.uint8)
        .transpose(0, 2, 3, 1)
    )  # (N, C, H, W) -> (N, H, W, C)
    func = lambda i: images_plot[i]
    draw_image(func, len(total_images))
    plt.savefig("original_images.png")


def pca_first(pca: PCA, features: np.ndarray[tuple[int, int], Any]):
    pca.fit(features)
    pca_features = fix_pca_sign(
        pca.transform(features)
    )  # pca は符号が逆転することもあるので、最初の成分が正になるように調整

    pca_features = minmax_scale(pca_features)

    # 画像の背景と前景を分離(マスク作成)
    bg_mask = pca_features[:, 0] < TH
    fg_mask = ~bg_mask
    return pca_features, fg_mask, bg_mask


def pca_second(pca, ret, total_features_for_pca, fg_mask, bg_mask):
    pca.fit(total_features_for_pca[fg_mask])
    pca_foreground = pca.transform(total_features_for_pca[fg_mask])
    pca_foreground = minmax_scale(pca_foreground)

    # 描画処理
    # for black background
    ret[bg_mask] = 0
    # new scaled foreground features
    ret[fg_mask] = pca_foreground


def get_src_img(folder_path: str):
    exts = ["jpg", "JPG", "png", "PNG"]
    ret = []
    for ext in exts:
        ret.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
    return ret


def main(args):
    _id = args.model_id
    model_dict = [
        ["dinov2_vits14", 384],
        ["dinov2_vitb14", 768],
        ["dinov2_vitl14", 1024],
        ["dinov2_vitg14", 1536],
    ]

    dinov2 = torch.hub.load("facebookresearch/dinov2", model_dict[_id][0]).to(DEVICE)
    feat_dim = model_dict[_id][1]
    patch_size = dinov2.patch_size  # patchsize=14
    folder_path = args.folder_path

    patch_h = PATCH_NUM
    patch_w = PATCH_NUM
    dim = 3

    image_size = patch_w * patch_size

    total_features = []
    original_images = []

    with torch.no_grad():
        for img_path in get_src_img(folder_path):
            features, transformed_image = get_features(dinov2, image_size, img_path)

            original_images.append(transformed_image)
            total_features.append(features)

    if len(total_features) == 0:
        print("No images found in the folder")
        return

    save_images(original_images)

    # total_featuresをPCAできるように改良
    num_images = len(total_features)
    total_features_for_pca = (
        torch.cat(total_features, dim=0)
        .cpu()
        .numpy()
        .reshape(num_images * patch_h * patch_w, feat_dim)
    )  # image_num(*H*w, 1024)

    pca = PCA(n_components=dim)

    pca_features, fg_mask, bg_mask = pca_first(pca, total_features_for_pca)
    # 一度目のPCA
    func = lambda i: pca_features[
        i * patch_h * patch_w : (i + 1) * patch_h * patch_w, 0
    ].reshape(patch_h, patch_w)  # noqa: E731

    draw_image(func, num_images)
    plt.savefig(f"first_pca_{model_dict[_id][0]}.png")

    # 二度目のPCA。反応した特徴量部分だけでPCAする
    ret = pca_features.copy()
    pca_second(pca, ret, total_features_for_pca, fg_mask, bg_mask)

    for_render = ret.reshape(num_images, patch_h, patch_w, dim)
    func = lambda i: for_render[i]
    draw_image(func, num_images)
    plt.savefig(f"result_{model_dict[_id][0]}.png")


def __entry_point():
    import argparse

    # explain program
    parser = argparse.ArgumentParser(description="PCA analysis of DINOv2 features.")
    parser.add_argument(
        "--model_id",
        type=int,
        default=2,
        help="Model ID (0: ViT-s/14, 1: ViT-b/14, 2: ViT-l/14, 3: ViT-g/14)",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="src_img/",
        help="Path to the folder containing images",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    __entry_point()
