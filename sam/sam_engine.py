import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 設定 ---
# 使用するモデルを選択 (tinyの方が軽い)
SAM2_MODEL_NAME = "facebook/sam2-hiera-tiny"
# SAM2_MODEL_NAME = "facebook/sam2-hiera-large"


# --- グローバル変数 (モデルインスタンス) ---
# FastAPI起動時にロードするためにグローバルに保持
class Sam:
    def __init__(self):
        """SAM2 Predictor のインスタンスを作成する (FastAPI起動時)"""
        device = torch.device("cuda")
        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_NAME)
        except:
            self.predictor = None

    def apply_mask_to_image(
        self, image_np_bgr, mask_np, random_color=False, borders=True
    ):
        """マスクを画像に重ね合わせる (OpenCVを使用 - この部分は変更なし)"""
        if mask_np.dtype != bool:
            mask_boolean = mask_np.astype(bool)
        else:
            mask_boolean = mask_np

        if random_color:
            color = np.array(
                [
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                ],
                dtype=np.uint8,
            )
        else:
            color = np.array([255, 144, 30], dtype=np.uint8)  # BGR: 青っぽい色

        h, w = mask_boolean.shape[-2:]
        mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        mask_colored[mask_boolean] = color

        alpha = 0.5
        blended = cv2.addWeighted(image_np_bgr, 1.0, mask_colored, alpha, 0)

        if borders:
            try:
                mask_uint8 = mask_boolean.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(blended, contours, -1, (255, 255, 255), thickness=2)
            except Exception as e:
                print(f"Warning: Failed to draw contours: {e}")

        return blended

    def predict_mask(
        self, image_path: str, input_points_list: list, input_labels_list: list
    ):
        # print(input_points_list)
        # print(input_labels_list)
        """指定された画像と点を使ってマスクを予測する (元のサンプルに近づける)"""
        if self.predictor is None:
            # load_model が FastAPI 起動時に呼ばれているはずだが、念のためチェック
            raise RuntimeError(
                "SAM2 predictor is not initialized. Application might not have started correctly."
            )

        try:
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img)
            image_np_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            input_point_np = np.array(input_points_list, dtype=np.float32)
            input_label_np = np.array(input_labels_list, dtype=np.int32)

            # --- 元のサンプルのデバイス設定、autocast/TF32設定を推論前に適用 ---

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(img_array)
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point_np,
                    point_labels=input_label_np,
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                if len(scores) == 0:
                    print("Warning: No masks predicted.")
                    # マスクが見つからなかった場合、元画像を返す
                    return image_np_bgr, 0.0

                # 最もスコアの高いマスクを選択
                best_mask_idx = np.argmax(scores)
                best_mask_np = masks[best_mask_idx]
                best_score = scores[best_mask_idx]

                # マスクを元画像(BGR)に適用
                result_image_np = self.apply_mask_to_image(
                    image_np_bgr, best_mask_np, borders=True
                )

                return result_image_np, float(best_score)

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None, None
        except Exception as e:
            print(f"Error during prediction in sam_engine: {e}")
            import traceback

            traceback.print_exc()
            return None, None
