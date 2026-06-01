from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import pandas as pd
import uuid
import hashlib
import random
import threading


# =========================
# パス設定
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

INDEX_HTML = BASE_DIR / "index.html"

PAIRS_CSV = DATA_DIR / "nail_near_pairs.csv"
RESULTS_CSV = DATA_DIR / "results.csv"


# =========================
# 結果CSVの列
# =========================

RESULT_COLUMNS = [
    "response_id",
    "timestamp",
    "user_id",
    "session_id",
    "pair_id",
    "left_file",
    "right_file",
    "choice_type",
    "chosen_file",
    "rejected_file",
    "reaction_time_ms",
    "distance",
]


# =========================
# アプリ初期化
# =========================

app = FastAPI()

# Render / GitHubでフォルダがない場合でも起動だけは落ちないようにする
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "images").mkdir(exist_ok=True)

# /static/images/xxx.png で画像を配信
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CSV同時書き込み対策
csv_lock = threading.Lock()


# =========================
# リクエストモデル
# =========================

class ChoiceRequest(BaseModel):
    user_id: str
    session_id: str
    pair_id: int

    left_file: str
    right_file: str

    # left, right, both_like, both_dislike, unsure, skip
    choice_type: str

    reaction_time_ms: int


# =========================
# ルート
# =========================

@app.get("/")
def index():
    """
    index.html は static の外、
    app.py と同じ階層に置く想定。
    """
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=f"index.html not found at {INDEX_HTML}"
        )

    return FileResponse(INDEX_HTML)


# =========================
# 内部関数
# =========================

def load_pairs_df() -> pd.DataFrame:
    """
    ペアCSVを読み込む。
    必須列:
    pair_id, file_1, file_2, distance
    """

    if not PAIRS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"nail_near_pairs.csv not found at {PAIRS_CSV}"
        )

    try:
        df = pd.read_csv(PAIRS_CSV)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read nail_near_pairs.csv: {str(e)}"
        )

    required_cols = ["pair_id", "file_1", "file_2", "distance"]

    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Required column missing: {col}. Current columns: {list(df.columns)}"
            )

    # 念のため欠損を除外
    df = df.dropna(subset=required_cols)

    return df


def get_answered_pair_ids(user_id: str) -> set[int]:
    """
    指定ユーザーがすでに回答したpair_id一覧を取得。
    """

    if not RESULTS_CSV.exists():
        return set()

    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except pd.errors.EmptyDataError:
        return set()
    except Exception:
        return set()

    if results_df.empty:
        return set()

    if "user_id" not in results_df.columns or "pair_id" not in results_df.columns:
        return set()

    user_results = results_df[results_df["user_id"] == user_id]

    return set(user_results["pair_id"].astype(int).tolist())


def stable_seed(text: str) -> int:
    """
    user_idごとに毎回同じランダム順になるようにする。
    """

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# =========================
# API
# =========================

@app.get("/api/pairs")
def get_pairs(
    user_id: str = Query(...),
    include_answered: bool = Query(False)
):
    """
    ペアデータを返す。
    - ユーザーごとに表示順をランダム化
    - 左右もユーザーごと・ペアごとにランダム化
    - 回答済みペアはデフォルトで除外
    """

    df = load_pairs_df()

    if not include_answered:
        answered_ids = get_answered_pair_ids(user_id)
        df = df[~df["pair_id"].isin(answered_ids)]

    records = df.to_dict(orient="records")

    # ユーザーごとに表示順を固定ランダム化
    rng_order = random.Random(stable_seed(f"order:{user_id}"))
    rng_order.shuffle(records)

    pairs = []

    for row in records:
        pair_id = int(row["pair_id"])
        file_1 = str(row["file_1"])
        file_2 = str(row["file_2"])

        # ユーザーごと・pair_idごとに左右を固定ランダム化
        rng_side = random.Random(stable_seed(f"side:{user_id}:{pair_id}"))

        if rng_side.random() < 0.5:
            left_file = file_1
            right_file = file_2
        else:
            left_file = file_2
            right_file = file_1

        pairs.append({
            "pair_id": pair_id,
            "left_file": left_file,
            "right_file": right_file,
            "left_image_url": f"/static/images/{left_file}",
            "right_image_url": f"/static/images/{right_file}",
            "distance": float(row["distance"]),
        })

    return {
        "total": len(pairs),
        "pairs": pairs
    }


@app.post("/api/choice")
def save_choice(choice: ChoiceRequest):
    """
    選択結果をCSVに追記。
    left/right の場合のみ chosen_file/rejected_file を入れる。
    both_like, both_dislike, unsure, skip は空欄。
    """

    allowed_choice_types = {
        "left",
        "right",
        "both_like",
        "both_dislike",
        "unsure",
        "skip",
    }

    if choice.choice_type not in allowed_choice_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid choice_type: {choice.choice_type}"
        )

    df = load_pairs_df()
    pair_df = df[df["pair_id"].astype(int) == int(choice.pair_id)]

    if pair_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"pair_id not found: {choice.pair_id}"
        )

    pair_row = pair_df.iloc[0]

    valid_files = {
        str(pair_row["file_1"]),
        str(pair_row["file_2"]),
    }

    if choice.left_file not in valid_files or choice.right_file not in valid_files:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Presented files do not match pair",
                "pair_id": choice.pair_id,
                "valid_files": list(valid_files),
                "left_file": choice.left_file,
                "right_file": choice.right_file,
            }
        )

    if choice.left_file == choice.right_file:
        raise HTTPException(
            status_code=400,
            detail="left_file and right_file are same"
        )

    chosen_file = ""
    rejected_file = ""

    if choice.choice_type == "left":
        chosen_file = choice.left_file
        rejected_file = choice.right_file

    elif choice.choice_type == "right":
        chosen_file = choice.right_file
        rejected_file = choice.left_file

    result = {
        "response_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": choice.user_id,
        "session_id": choice.session_id,
        "pair_id": choice.pair_id,
        "left_file": choice.left_file,
        "right_file": choice.right_file,
        "choice_type": choice.choice_type,
        "chosen_file": chosen_file,
        "rejected_file": rejected_file,
        "reaction_time_ms": choice.reaction_time_ms,
        "distance": float(pair_row["distance"]),
    }

    with csv_lock:
        result_df = pd.DataFrame([result], columns=RESULT_COLUMNS)

        if RESULTS_CSV.exists():
            result_df.to_csv(
                RESULTS_CSV,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig"
            )
        else:
            result_df.to_csv(
                RESULTS_CSV,
                index=False,
                encoding="utf-8-sig"
            )

    return {
        "status": "ok",
        "saved": result
    }


@app.get("/api/progress")
def get_progress(user_id: str = Query(...)):
    """
    進捗確認用。
    """

    pairs_df = load_pairs_df()
    total_pairs = len(pairs_df)

    answered_ids = get_answered_pair_ids(user_id)
    answered_count = len(answered_ids)

    return {
        "user_id": user_id,
        "total_pairs": total_pairs,
        "answered_pairs": answered_count,
        "remaining_pairs": max(total_pairs - answered_count, 0),
    }


@app.get("/api/results")
def get_results():
    """
    保存済み結果をJSONで確認。
    """

    if not RESULTS_CSV.exists():
        return {
            "total": 0,
            "results": []
        }

    try:
        df = pd.read_csv(RESULTS_CSV)
    except pd.errors.EmptyDataError:
        return {
            "total": 0,
            "results": []
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read results.csv: {str(e)}"
        )

    return {
        "total": len(df),
        "results": df.to_dict(orient="records")
    }


@app.get("/api/results/download")
def download_results():
    """
    結果CSVをダウンロード。
    """

    if not RESULTS_CSV.exists():
        raise HTTPException(
            status_code=404,
            detail="results.csv not found"
        )

    return FileResponse(
        RESULTS_CSV,
        media_type="text/csv",
        filename="nail_choice_results.csv"
    )


@app.get("/api/debug")
def debug():
    """
    Render上でファイル配置を確認するためのデバッグ用。
    """

    return {
        "base_dir": str(BASE_DIR),
        "index_html": str(INDEX_HTML),
        "index_html_exists": INDEX_HTML.exists(),

        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "data_files": [p.name for p in DATA_DIR.glob("*")] if DATA_DIR.exists() else [],

        "pairs_csv": str(PAIRS_CSV),
        "pairs_csv_exists": PAIRS_CSV.exists(),

        "static_dir": str(STATIC_DIR),
        "static_dir_exists": STATIC_DIR.exists(),
        "static_files": [p.name for p in STATIC_DIR.glob("*")] if STATIC_DIR.exists() else [],

        "images_dir": str(STATIC_DIR / "images"),
        "images_dir_exists": (STATIC_DIR / "images").exists(),
        "image_files_sample": [
            p.name for p in (STATIC_DIR / "images").glob("*")
        ][:20] if (STATIC_DIR / "images").exists() else [],
    }


# =========================
# python app.py で起動された場合にも動くようにする
# =========================

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )