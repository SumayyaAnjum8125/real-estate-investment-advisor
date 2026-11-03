"""Model loading, feature building and prediction helpers for the Streamlit app.

Nothing in here imports Streamlit, so it can be tested on its own.
The feature engineering below mirrors Steps 3-8 of the notebook exactly:
the models were trained on those columns, so any drift here silently
degrades predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --- Paths (app/ sits next to models/ and data/) -----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "cleaned_housing.csv"

# --- Constants copied from the notebook --------------------------------------
AMENITIES = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}
CAT_COLS = ["City", "Property_Type", "Furnished_Status", "Facing",
            "Owner_Type", "Availability_Status"]
NUM_REF_COLS = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
                "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals"]
GOOD_GROWTH_THRESHOLD = 0.09   # Good_Investment = Growth_Rate >= 9 % / year
HORIZON_YEARS = 5
LAKH = 100_000                 # 1 lakh = 100,000 rupees


# --- Loading -----------------------------------------------------------------
def load_models():
    """Return (classifier_pipeline, regressor_pipeline, feature_cols)."""
    clf = joblib.load(MODELS_DIR / "best_classifier.joblib")
    reg = joblib.load(MODELS_DIR / "best_regressor.joblib")
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return clf, reg, feature_cols


def load_reference() -> dict:
    """Read the cleaned dataset once and derive what the UI needs:
    dropdown options, slider ranges and per-city price-per-sq-ft values."""
    df = pd.read_csv(DATA_PATH, usecols=CAT_COLS + NUM_REF_COLS)
    df["PPSF_Rs"] = df["Price_in_Lakhs"] * LAKH / df["Size_in_SqFt"]

    options = {c: sorted(df[c].dropna().astype(str).unique()) for c in CAT_COLS}
    ranges = {
        c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
        for c in NUM_REF_COLS
    }
    return {
        "options": options,
        "ranges": ranges,
        "city_ppsf": df[["City", "PPSF_Rs"]],
    }


# --- Feature building --------------------------------------------------------
def build_features(form: dict, feature_cols: list[str]) -> pd.DataFrame:
    """Turn raw form values into the one-row frame the pipelines expect."""
    floor_no = int(form["Floor_No"])
    total_floors = max(int(form["Total_Floors"]), 1)

    row = {
        "BHK": int(form["BHK"]),
        "Size_in_SqFt": float(form["Size_in_SqFt"]),
        "Price_in_Lakhs": float(form["Price_in_Lakhs"]),
        "Year_Built": int(form["Year_Built"]),
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Floor_Ratio": floor_no / total_floors,
        "Nearby_Schools": int(form["Nearby_Schools"]),
        "Nearby_Hospitals": int(form["Nearby_Hospitals"]),
        "Transport_Score": TRANSPORT_MAP[form["Public_Transport_Accessibility"]],
        "Parking_Space": int(bool(form["Parking_Space"])),
        "Security": int(bool(form["Security"])),
        "Floor_Invalid": 0,  # the form blocks floor > total floors
        "City": form["City"],
        "Property_Type": form["Property_Type"],
        "Furnished_Status": form["Furnished_Status"],
        "Facing": form["Facing"],
        "Owner_Type": form["Owner_Type"],
        "Availability_Status": form["Availability_Status"],
    }
    for a in AMENITIES:
        row["Has_" + a] = int(a in form["Amenities"])

    return pd.DataFrame([row])[feature_cols]


# --- Prediction --------------------------------------------------------------
@dataclass
class Prediction:
    p_good: float          # P(Good_Investment = 1)
    is_good: bool
    confidence: float      # probability of the predicted class (always >= 0.5)
    price_now: float       # lakhs
    price_5y: float        # lakhs
    total_gain: float      # fraction, e.g. 0.52 = +52 %
    cagr: float            # implied constant annual growth


def predict(clf, reg, X: pd.DataFrame) -> Prediction:
    p_good = float(clf.predict_proba(X)[0, 1])
    is_good = p_good >= 0.5
    price_now = float(X["Price_in_Lakhs"].iloc[0])
    price_5y = float(reg.predict(X)[0])
    total_gain = price_5y / price_now - 1
    cagr = (price_5y / price_now) ** (1 / HORIZON_YEARS) - 1 if price_5y > 0 else float("nan")
    return Prediction(
        p_good=p_good,
        is_good=is_good,
        confidence=p_good if is_good else 1 - p_good,
        price_now=price_now,
        price_5y=price_5y,
        total_gain=total_gain,
        cagr=cagr,
    )


def price_what_if(clf, X: pd.DataFrame, lo: float = 0.6, hi: float = 1.4, n: int = 21) -> pd.DataFrame:
    """P(good) when only the asking price changes (everything else fixed)."""
    mult = np.linspace(lo, hi, n)
    grid = pd.concat([X] * n, ignore_index=True)
    grid["Price_in_Lakhs"] = X["Price_in_Lakhs"].iloc[0] * mult
    return pd.DataFrame({
        "Price_in_Lakhs": grid["Price_in_Lakhs"],
        "p_good": clf.predict_proba(grid)[:, 1],
    })


def projection_path(price_now: float, cagr: float) -> pd.DataFrame:
    years = np.arange(0, HORIZON_YEARS + 1)
    return pd.DataFrame({"Year": years, "Price_in_Lakhs": price_now * (1 + cagr) ** years})


def city_context(ref: dict, city: str, price_lakhs: float, size_sqft: float) -> dict:
    """Where this property's price/sq ft sits among listings in the same city."""
    vals = ref["city_ppsf"].loc[ref["city_ppsf"]["City"] == city, "PPSF_Rs"].to_numpy()
    ppsf = price_lakhs * LAKH / size_sqft
    median = float(np.median(vals))
    return {
        "values": vals,
        "ppsf": ppsf,
        "median": median,
        "vs_median": ppsf / median - 1,
        "percentile": float((vals < ppsf).mean() * 100),
    }


# --- Formatting --------------------------------------------------------------
def fmt_lakhs(x: float) -> str:
    """₹ 85.5 L  /  ₹ 1.52 Cr  (1 crore = 100 lakhs)."""
    return f"₹{x / 100:,.2f} Cr" if abs(x) >= 100 else f"₹{x:,.1f} L"
