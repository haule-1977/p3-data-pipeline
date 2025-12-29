from __future__ import annotations

import glob
import os
import re
from datetime import timedelta
import pandas as pd

# Script start indicator (helps debugging)
print("✅ p3data.py started")

# =========================
# CONFIG SECTION
# =========================
# INPUT_FOLDER = "." means:
# 👉 Look for Excel files in the SAME folder as this script
INPUT_FOLDER = "."

# Output master file name (will be created/overwritten)
OUTPUT_FILE = "P3_MASTER_VERIFIED.xlsx"

# Assumption:
# On average, a pregnant mom (S1) has ~20 weeks remaining
PREDICT_REMAINING_WEEKS = 20

# Fake score threshold:
# <= 70 → usable for Sales
DEFAULT_FAKE_THRESHOLD = 70


# =========================
# HELPER FUNCTIONS
# =========================
def normalize_phone(x) -> str:
    """
    Normalize phone numbers so we can:
    - Detect duplicates reliably
    - Detect fake / invalid numbers

    Logic:
    - Convert NaN → empty string
    - Remove trailing .0 from Excel numeric phones
    - Remove all non-digit characters
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)   # remove ".0"
    s = re.sub(r"\D+", "", s)    # keep digits only
    return s


def build_register_date(df: pd.DataFrame) -> pd.Series:
    """
    Build a reliable Register_Date for each record.

    Priority:
    1️⃣ Use Year + Month.1 + Day columns (most accurate)
    2️⃣ Fallback to 'Date' column if needed
    """

    # Preferred method: separate Year / Month / Day
    if {"Year", "Month.1", "Day"}.issubset(df.columns):
        s = pd.to_datetime(
            df[["Year", "Month.1", "Day"]].rename(
                columns={"Year": "year", "Month.1": "month", "Day": "day"}
            ),
            errors="coerce"
        )

        # If most rows parsed successfully, use it
        if s.isna().mean() < 0.9:
            return s

    # Fallback: parse "Date" column (Vietnam format dd/mm/yyyy)
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    # If everything fails, return NaT
    return pd.Series([pd.NaT] * len(df), index=df.index)


def pregnancy_status(row: pd.Series) -> str:
    """
    Determine pregnancy status using business rules:

    - S1 present → Pregnant mom
    - S2 present → Mom with child
    - Otherwise → Unknown
    """
    if pd.notna(row.get("S1")):
        return "Pregnant"
    if pd.notna(row.get("S2")):
        return "Has_Child"
    return "Unknown"


# =========================
# MAIN PROCESS
# =========================
def main():
    print("📂 Working directory:", os.getcwd())

    # Find all Excel files in the folder
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.xlsx"))

    # Exclude output file if script was run before
    files = [f for f in files if os.path.basename(f) != OUTPUT_FILE]

    print(f"📄 Found {len(files)} Excel file(s)")
    for f in files:
        print(" -", os.path.basename(f))

    # Stop if no input files found
    if not files:
        print("❌ No Excel files found. Stop.")
        return

    # =========================
    # MERGE ALL FILES
    # =========================
    frames = []
    for f in files:
        print("📥 Reading:", os.path.basename(f))
        df = pd.read_excel(f, engine="openpyxl")
        df["Source_File"] = os.path.basename(f)  # track data origin
        frames.append(df)

    # Merge all data into one DataFrame
    data = pd.concat(frames, ignore_index=True)

    # =========================
    # DATA NORMALIZATION
    # =========================

    # Normalize phone numbers
    data["Mobile_norm"] = data["Mobile"].apply(normalize_phone)

    # Build registration date
    data["Register_Date"] = build_register_date(data)

    # Determine pregnancy status
    data["Pregnancy_Status"] = data.apply(pregnancy_status, axis=1)

    today = pd.Timestamp.today().normalize()

    # =========================
    # PREDICTIONS
    # =========================

    # Predict giving birth date (for pregnant moms)
    data["Predicted_Giving_Birth_Date"] = pd.NaT
    mask_preg = data["Pregnancy_Status"] == "Pregnant"
    data.loc[mask_preg, "Predicted_Giving_Birth_Date"] = (
        data.loc[mask_preg, "Register_Date"]
        + pd.to_timedelta(PREDICT_REMAINING_WEEKS * 7, unit="D")
    )

    # Predict child age (months) for moms with child
    data["Child_Age_Months"] = pd.NA
    mask_child = data["Pregnancy_Status"] == "Has_Child"
    data.loc[mask_child, "Child_Age_Months"] = (
        (today - data.loc[mask_child, "Register_Date"]) / timedelta(days=30)
    ).round(1)

    # =========================
    # FAKE SCORE LOGIC (0–100)
    # =========================
    score = 0

    # Missing first name → suspicious
    score += data["FirstName"].isna().astype(int) * 20

    # Invalid phone length
    score += (data["Mobile_norm"].str.len() < 9).astype(int) * 25

    # Duplicate phone numbers
    score += data.duplicated("Mobile_norm", keep=False).astype(int) * 20

    # Missing registration date
    score += data["Register_Date"].isna().astype(int) * 15

    # Unrealistic child age (> 6 years)
    score += (data["Child_Age_Months"].fillna(0) > 72).astype(int) * 20

    # Final fake score (capped at 100)
    data["Fake_Score"] = score.clip(0, 100)

    # Confidence labels for business use
    data["Confidence_Label"] = pd.cut(
        data["Fake_Score"],
        bins=[-1, 30, 70, 100],
        labels=["High", "Medium", "Low"]
    )

    # Simple sales-ready flag
    data["Sales_Ready"] = data["Fake_Score"] <= DEFAULT_FAKE_THRESHOLD

    # Sort best data first
    data = data.sort_values(["Fake_Score", "Register_Date"])

    # =========================
    # EXPORT
    # =========================
    data.to_excel(OUTPUT_FILE, index=False)

    print("✅ DONE!")
    print("📁 Output:", OUTPUT_FILE)


# Entry point
if __name__ == "__main__":
    main()
