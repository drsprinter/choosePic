from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import uuid
import hashlib
import random
import threading


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

PAIRS_CSV = DATA_DIR / "nail_near_pairs.csv"
RESULTS_CSV = DATA_DIR / "results.csv"

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

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

csv_lock = threading.Lock()


class ChoiceRequest(BaseModel):
    user_id: str
    session_id: str
    pair_id: int

    left_file: str
    right_file: str

    # left, right, both_like, both_dislike, unsure, skip
    choice_type: str

    reaction_time_ms: int


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def load_pairs_df() -> pd.DataFrame:
    if not PAIRS_CSV.exists():
        raise HTTPException(status_code=404, detail="nail_near_pairs.csv not found")

    df = pd.read_csv(PAIRS_CSV)

    required_cols = ["pair_id", "file_1", "file_2", "distance"]
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Required column missing: {col}"
            )

    return df


def get_answered_pair_ids(user_id: str) -> set[int]:
    if not RESULTS_CSV.exists():
        return set()

    try:
        results_df = pd.read_csv(RESULTS_CSV)
    except pd.errors.EmptyDataError:
        return set()

    if results_df.empty:
        return set()

    user_results = results_df[results_df["user_id"] == user_id]

    return set(user_results["pair_id"].astype(int).tolist())


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


@app.get("/api/pairs")
def get_pairs(
    user_id: str = Query(...),
    include_answered: bool = Query(False)
):
    """
    ユーザーごとにランダム化されたペア一覧を返す。
    すでに回答済みのpair_idはデフォルトで除外。
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

        # ペアごと・ユーザーごとに左右を固定ランダム化
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
    left / right の場合のみ chosen_file / rejected_file を決定。
    それ以外は chosen_file / rejected_file を空欄にする。
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
        raise HTTPException(status_code=400, detail="Invalid choice_type")

    df = load_pairs_df()
    pair_df = df[df["pair_id"] == choice.pair_id]

    if pair_df.empty:
        raise HTTPException(status_code=404, detail="pair_id not found")

    pair_row = pair_df.iloc[0]
    valid_files = {str(pair_row["file_1"]), str(pair_row["file_2"])}

    if choice.left_file not in valid_files or choice.right_file not in valid_files:
        raise HTTPException(status_code=400, detail="Presented files do not match pair")

    if choice.left_file == choice.right_file:
        raise HTTPException(status_code=400, detail="left_file and right_file are same")

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

    DATA_DIR.mkdir(exist_ok=True)

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

    return {
        "total": len(df),
        "results": df.to_dict(orient="records")
    }


@app.get("/api/results/download")
def download_results():
    if not RESULTS_CSV.exists():
        raise HTTPException(status_code=404, detail="results.csv not found")

    return FileResponse(
        RESULTS_CSV,
        media_type="text/csv",
        filename="nail_choice_results.csv"
    )