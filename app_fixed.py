"""
Riyadh Restaurants Chatbot (FastAPI)

Features:
- Loads & cleans Kaggle CSV dataset
- Custom NLP pipeline to parse user queries:
  * top-K (e.g., "أفضل 10", "top 3")
  * rating threshold (e.g., "فوق 4", ">=4.2")
  * price intent (cheap/moderate/expensive) with robust matching
  * area / neighborhood (supports: "حي X", "في X", "بالـ X")
  * road/street phrase support: "طريق الملك عبدالله", "شارع التحلية" (address filter)
  * region hints: شمال/جنوب/شرق/غرب (approximate by lat/lng quantiles)
  * category / cuisine (Arabic/English keywords)
  * nearest intent using (lat,lng)
- Answers are ALWAYS data-driven (no generic text)
- Optional LLM + (light RAG) using LangChain + OpenAI if you set env vars:
    OPENAI_API_KEY, USE_LLM_RAG=1
  If not set, the bot runs fully offline with the custom NLP pipeline.

Run (PowerShell):
  $env:CSV_PATH="data\riyadh_restaurants_clean_exported.csv"
  uvicorn app_fixed:app --reload --reload-dir .
"""

from __future__ import annotations

import os
import re
import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# Data Processing
# =========================

REQUIRED_COLUMNS = {"name", "categories", "address", "lat", "lng", "price", "rating"}


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop_duplicates()

    # Ensure required fields exist
    df = df.dropna(subset=["name", "lat", "lng"])

    # Normalize text columns
    for col in ["name", "categories", "address", "price"]:
        df[col] = df[col].astype(str).str.strip()

    # Lowercase helpers for search
    df["name_l"] = df["name"].str.lower()
    df["categories_l"] = df["categories"].str.lower()
    df["address_l"] = df["address"].str.lower()
    df["price_l"] = df["price"].str.lower()

    # Rating numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    # Coerce lat/lng numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df = df.dropna(subset=["lat", "lng"])

    return df.reset_index(drop=True)


# =========================
# Custom NLP Pipeline
# =========================

PRICE_KEYWORDS = {
    "رخيص": ["cheap", "budget", "inexpensive", "$", "low"],
    "رخيصة": ["cheap", "budget", "inexpensive", "$", "low"],
    "متوسط": ["moderate", "mid", "average", "$$"],
    "متوسطه": ["moderate", "mid", "average", "$$"],
    "غالي": ["expensive", "luxury", "premium", "$$$", "high"],
    "غالية": ["expensive", "luxury", "premium", "$$$", "high"],
}

CAT_KEYWORDS = {
    # Arabic
    "كافيه": ["cafe", "coffee"],
    "مقهى": ["cafe", "coffee"],
    "قهوة": ["cafe", "coffee"],
    "لبناني": ["lebanese"],
    "ايطالي": ["italian"],
    "إيطالي": ["italian"],
    "سوشي": ["sushi", "japanese"],
    "ياباني": ["japanese", "sushi"],
    "برجر": ["burger"],
    "بيتزا": ["pizza"],
    "ستيك": ["steak"],
    "هندي": ["indian"],
    "صيني": ["chinese"],
    "تركي": ["turkish"],
    "مكسيكي": ["mexican"],
    # English
    "cafe": ["cafe", "coffee"],
    "coffee": ["cafe", "coffee"],
    "fine dining": ["fine dining", "fine-dining", "finedining"],
}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def extract_top_k(text: str, default: int = 5, max_k: int = 20) -> int:
    t = normalize(text)
    m = re.search(r"(?:top|أفضل|افضل)\s*(\d{1,2})", t, flags=re.I)
    if m:
        k = int(m.group(1))
        return max(1, min(max_k, k))

    m2 = re.search(r"\b(\d{1,2})\s*(?:مطاعم|مطعم|أماكن|مكان|restaurants?)\b", t, flags=re.I)
    if m2:
        k = int(m2.group(1))
        return max(1, min(max_k, k))

    return default


def extract_rating_threshold(text: str) -> Optional[float]:
    t = normalize(text)
    m = re.search(r"(?:فوق|أكثر\s+من|اقل\s+من|under|below|above|>=|<=|>|<|rating)\s*([0-5](?:\.\d)?)", t, flags=re.I)
    if not m:
        m = re.search(r"\b([0-5](?:\.\d)?)\s*\+\b", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_lat_lng(text: str) -> Optional[Tuple[float, float]]:
    t = text.replace("(", " ").replace(")", " ").replace("،", ",")
    nums = re.findall(r"(-?\d{1,3}\.\d+)", t)
    if len(nums) >= 2:
        lat = float(nums[0])
        lng = float(nums[1])
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    return None


def extract_category(text: str) -> Optional[str]:
    t = normalize(text)
    for k, variants in CAT_KEYWORDS.items():
        if normalize(k) in t:
            return variants[0]  # token likely present in categories_l
    return None


def extract_price_token(text: str) -> Optional[str]:
    t = normalize(text)
    for ar, tokens in PRICE_KEYWORDS.items():
        if normalize(ar) in t:
            return tokens[0]

    if "cheap" in t or "budget" in t:
        return "cheap"
    if "moderate" in t or "mid" in t:
        return "moderate"
    if "expensive" in t or "luxury" in t or "premium" in t:
        return "expensive"

    return None


def extract_region(text: str) -> Optional[str]:
    t = normalize(text)
    if "شمال" in t or "north" in t:
        return "north"
    if "جنوب" in t or "south" in t:
        return "south"
    if "شرق" in t or "east" in t:
        return "east"
    if "غرب" in t or "west" in t:
        return "west"
    return None


def extract_area(text: str) -> Optional[str]:
    """
    Supports:
      - "حي الياسمين"
      - "في الياسمين"
      - "بالملقا" / "بالمحمدية"
      - road phrases: "طريق الملك عبدالله", "شارع التحلية" (used as address filter)
    Returns a best-effort keyword phrase for address matching.
    """
    t = text.strip()
    stop = {"طريق", "شارع", "road", "street", "st", "rd"}

    # Road/street phrase (up to 3 words after)
    m = re.search(r"(?:طريق|شارع)\s+([^\s،,.?!]+(?:\s+[^\s،,.?!]+){0,2})", t)
    if m:
        prefix = "طريق" if "طريق" in m.group(0) else "شارع"
        phrase = f"{prefix} {m.group(1)}".strip()
        # avoid returning generic single-word stop if phrase is too short
        if normalize(phrase) not in stop:
            return phrase

    # حي X
    m = re.search(r"حي\s+([^\s،,.?!]+)", t)
    if m:
        cand = m.group(1)
        if normalize(cand) not in stop:
            return cand

    # في X
    m = re.search(r"\bفي\s+([^\s،,.?!]+)", t)
    if m:
        cand = m.group(1)
        if cand.lower() not in {"الرياض", "riyadh"} and normalize(cand) not in stop:
            return cand

    # بالـ X / بالX
    m = re.search(r"\bبال(?:ـ)?\s*([^\s،,.?!]+)", t)
    if m:
        cand = m.group(1)
        if cand.lower() not in {"الرياض", "riyadh"} and normalize(cand) not in stop:
            return cand

    return None


def format_row(r: pd.Series) -> str:
    return (
        f"- {r['name']} | {r['categories']} | {r['address']} | "
        f"Price: {r['price']} | Rating: {float(r['rating']):.1f}"
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def match_price_df(df: pd.DataFrame, intent: str) -> pd.DataFrame:
    """
    Map semantic intent -> dataset price patterns.
    Tries multiple patterns because datasets vary ($, $$, cheap, moderate, etc).
    """
    intent = normalize(intent)
    if intent in {"cheap", "low"}:
        patterns = ["cheap", "budget", "inexpensive", "$"]
    elif intent in {"moderate", "mid"}:
        patterns = ["moderate", "mid", "average", "$$"]
    elif intent in {"expensive", "high"}:
        patterns = ["expensive", "luxury", "premium", "$$$"]
    else:
        patterns = [intent]

    mask = False
    for p in patterns:
        mask = mask | df["price_l"].str.contains(re.escape(p), na=False)
    return df[mask]


# =========================
# Optional LLM + (Light) RAG (LangChain + OpenAI)
# =========================

def llm_rag_answer(question: str, df: pd.DataFrame) -> Optional[str]:
    """
    Optional LLM polishing grounded on retrieved rows.
    Enabled only if:
      USE_LLM_RAG=1 and OPENAI_API_KEY is set and deps installed.
    """
    if os.getenv("USE_LLM_RAG", "0") != "1":
        return None
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except Exception:
        return None

    q = normalize(question)
    tmp = df.copy()

    # Simple retrieval-style heuristic
    key = q[:20] if len(q) >= 3 else q
    tmp["hit"] = (
        tmp["name_l"].str.contains(re.escape(key), na=False) |
        tmp["categories_l"].str.contains(re.escape(key), na=False) |
        tmp["address_l"].str.contains(re.escape(key), na=False)
    )
    cand = tmp[tmp["hit"]].sort_values("rating", ascending=False).head(8)
    if cand.empty:
        cand = df.sort_values("rating", ascending=False).head(8)

    context = "\n".join(format_row(r) for _, r in cand.iterrows())

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a data-grounded assistant. Answer ONLY using the provided dataset context. "
         "If the context is insufficient, say you couldn't find it in the data."),
        ("user",
         "Question: {question}\n\nDataset context:\n{context}\n\n"
         "Answer in Arabic, and keep it concise.")
    ])

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    out = llm.invoke(prompt.format_messages(question=question, context=context))
    return getattr(out, "content", None) or str(out)


# =========================
# Chatbot Engine
# =========================

class RestaurantBot:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _is_nearest_intent(self, q: str) -> bool:
        qn = normalize(q)
        return any(w in qn for w in ["أقرب", "اقرب", "nearest", "near me", "قريب"])

    def _apply_filters(self, question: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        q = normalize(question)
        df = self.df.copy()
        meta: Dict[str, Any] = {}

        # top-k
        meta["top_k"] = extract_top_k(q)

        # area / address phrase
        area = extract_area(question)
        if area:
            meta["area"] = area
            df = df[df["address_l"].str.contains(normalize(area), na=False)]

        # category / cuisine
        cat_token = extract_category(question)
        if cat_token:
            meta["category"] = cat_token
            df = df[df["categories_l"].str.contains(normalize(cat_token), na=False)]

        # rating threshold
        thr = extract_rating_threshold(question)
        if thr is not None:
            meta["rating_min"] = thr
            if re.search(r"(?:اقل\s+من|under|below|<=|<)", q, flags=re.I):
                df = df[df["rating"] <= thr]
            else:
                df = df[df["rating"] >= thr]

        # price (robust)
        price_token = extract_price_token(question)
        if price_token:
            meta["price"] = price_token
            df = match_price_df(df, price_token)

        # fine dining heuristic
        if re.search(r"\bfine\s*dining\b", q) or "فاين" in question:
            meta["fine_dining"] = True
            if "category" not in meta:
                df = df[df["rating"] >= self.df["rating"].quantile(0.90)]
            if "price" not in meta:
                df = match_price_df(df, "expensive")

        # region (north/south/east/west) approximate by lat/lng quantiles
        region = extract_region(question)
        if region:
            meta["region"] = region
            lat_q25 = self.df["lat"].quantile(0.25)
            lat_q75 = self.df["lat"].quantile(0.75)
            lng_q25 = self.df["lng"].quantile(0.25)
            lng_q75 = self.df["lng"].quantile(0.75)

            if region == "north":
                df = df[df["lat"] >= lat_q75]
            elif region == "south":
                df = df[df["lat"] <= lat_q25]
            elif region == "east":
                df = df[df["lng"] >= lng_q75]
            elif region == "west":
                df = df[df["lng"] <= lng_q25]

        return df, meta

    def answer(self, question: str) -> Dict[str, Any]:
        base_df, meta = self._apply_filters(question)
        user_loc = extract_lat_lng(question)
        k = int(meta.get("top_k", 5))

        # If user asks for 'nearest' but didn't provide coordinates, ask for them.
        if self._is_nearest_intent(question) and user_loc is None:
            return {
                "ok": False,
                "message": (
                    "عشان أطلع (الأقرب) بدقة لازم ترسل موقعك بصيغة (lat, lng) مثل: 24.744, 46.636. "
                    "أو إذا تبين فلترة على طريق/شارع بدون موقع، اكتب: 'مطاعم ايطالية على طريق الملك عبدالله'."
                ),
                "meta": meta,
            }

        # Nearest path (distance-based)
        if self._is_nearest_intent(question) and user_loc is not None:
            df = base_df if not base_df.empty else self.df
            lat_u, lng_u = user_loc
            tmp = df.copy()
            tmp["distance_km"] = tmp.apply(
                lambda r: haversine_km(lat_u, lng_u, float(r["lat"]), float(r["lng"])),
                axis=1
            )
            out = tmp.sort_values("distance_km", ascending=True).head(k)
            if out.empty:
                return {"ok": False, "message": "ما لقيت نتائج قريبة بناءً على بيانات الملف.", "meta": meta}

            lines = [f"أقرب {len(out)} نتيجة حسب موقعك (lat,lng):"]
            for _, r in out.iterrows():
                lines.append(
                    f"- {r['name']} | {r['categories']} | {r['address']} | "
                    f"{float(r['distance_km']):.2f} km | Rating: {float(r['rating']):.1f}"
                )
            msg = "\n".join(lines)
            llm_msg = llm_rag_answer(question, out)
            return {"ok": True, "message": llm_msg or msg, "meta": meta}

        # Ranked path (top rated within filters)
        df = base_df
        if df.empty:
            top = self.df.sort_values("rating", ascending=False).head(k)
            lines = ["ما لقيت نتائج تطابق الفلاتر في سؤالك. هذه أفضل النتائج العامة من البيانات:"]
            for _, r in top.iterrows():
                lines.append(format_row(r))
            msg = "\n".join(lines)
            llm_msg = llm_rag_answer(question, top)
            return {"ok": True, "message": llm_msg or msg, "meta": {**meta, "fallback": True}}

        top = df.sort_values("rating", ascending=False).head(k)
        lines = [f"أفضل {len(top)} نتيجة من البيانات:"]
        for _, r in top.iterrows():
            lines.append(format_row(r))
        msg = "\n".join(lines)

        llm_msg = llm_rag_answer(question, top)
        return {"ok": True, "message": llm_msg or msg, "meta": meta}


# =========================
# FastAPI App
# =========================

CSV_PATH = os.getenv("CSV_PATH", "data/riyadh_restaurants_clean_exported.csv")

try:
    DF = load_and_clean(CSV_PATH)
except Exception as e:
    DF = pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["name_l", "categories_l", "address_l", "price_l"])
    LOAD_ERROR = str(e)
else:
    LOAD_ERROR = None

bot = RestaurantBot(DF)

app = FastAPI(title="Riyadh Restaurants Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    return bot.answer(req.question)


@app.get("/health")
def health():
    return {"ok": LOAD_ERROR is None, "load_error": LOAD_ERROR, "rows": int(len(DF))}
