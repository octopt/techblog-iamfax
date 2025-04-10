import base64
import io
import os
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

# SAMエンジンモジュールをインポート
from sam_engine import Sam

# --- 定数 ---
UPLOAD_DIR = "backend/uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

sam = None


# --- モデルライフサイクル管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリケーション起動時にモデルをロード
    # load_model()
    global sam
    sam = Sam()
    # # CUDAキャッシュのクリア (必要な場合)
    # if DEVICE.type == 'cuda':
    #     import torch
    #     torch.cuda.empty_cache()
    yield
    # アプリケーション終了時のクリーンアップ (もしあれば)
    print("Application shutdown.")


app = FastAPI(lifespan=lifespan)

# --- 静的ファイルとテンプレートの設定 ---
# app.mount("/static", StaticFiles(directory="frontend/static"), name="static") # 別途CSS/JSファイルを用意する場合
templates = Jinja2Templates(directory="frontend")


# --- リクエストボディのモデル定義 ---
class SegmentationRequest(BaseModel):
    filename: str
    points: list[list[float]]  # [[x1, y1], [x2, y2], ...]
    labels: list[int]  # [1, 0, ...] (1: positive, 0: negative)


# --- ルート ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """フロントエンドのHTMLを返す"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """画像をアップロードし、一時ファイル名を返す"""
    try:
        # ファイル拡張子をチェック (例: jpeg, png)
        allowed_extensions = {"jpg", "jpeg", "png"}
        extension = file.filename.split(".")[-1].lower()
        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only JPG, JPEG, PNG allowed.",
            )

        # ユニークなファイル名を生成
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.{extension}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        # ファイルを保存
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        # 画像サイズを取得して返す (フロントでの座標計算用)
        img = Image.open(io.BytesIO(contents))
        width, height = img.size

        return JSONResponse(
            content={"filename": filename, "width": width, "height": height}
        )

    except Exception as e:
        print(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image.")


@app.post("/segment/")
async def segment_image(request_data: SegmentationRequest):
    """画像とポイントを受け取り、セグメンテーション結果を返す"""
    filepath = os.path.join(UPLOAD_DIR, request_data.filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found on server.")

    if (
        not request_data.points
        or not request_data.labels
        or len(request_data.points) != len(request_data.labels)
    ):
        raise HTTPException(
            status_code=400, detail="Invalid points or labels provided."
        )

    try:
        # ポイントデータをNumpy配列に変換
        input_points_np = np.array(request_data.points, dtype=np.float32)
        input_labels_np = np.array(request_data.labels, dtype=np.int32)

        # SAMエンジンで予測を実行
        global sam
        result_image_np, score = sam.predict_mask(
            filepath, input_points_np.tolist(), input_labels_np.tolist()
        )  # SAMエンジンに合わせてリスト形式で渡す

        if result_image_np is None:
            raise HTTPException(status_code=500, detail="Segmentation failed.")

        # 結果画像をBase64エンコードして返す
        _, buffer = cv2.imencode(".jpg", result_image_np)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return JSONResponse(
            content={
                "result_image": f"data:image/jpeg;base64,{img_base64}",
                "score": float(score) if score is not None else None,
            }
        )

    except Exception as e:
        print(f"Error during segmentation API call: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Segmentation failed.")


# --- アプリケーション実行 (デバッグ用) ---
if __name__ == "__main__":
    import uvicorn

    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # reload=True は開発時に便利だが、モデルの再ロードが発生する可能性あり
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
