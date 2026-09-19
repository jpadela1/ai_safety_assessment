"""
dataset_specs.py — the four datasets, each with its own context and its own
physical-safety specifications.

EDIT THIS FILE, not the notebooks, when you want to change a context, a
plausible-value range, a safety-critical stratum, or a protected attribute.
Everything downstream (which sub-dimensions apply, which are N/A and why,
which injectors are legal) is derived from what is written here.

Contexts available: health, loan_finance, transportation, biology, chemistry,
academic, general.
"""
from __future__ import annotations

from safety_lib import DATA, DatasetSpec

# --------------------------------------------------------------------------- #
# 1. Diabetes 130-US hospitals  —  context: health, modality: tabular
# --------------------------------------------------------------------------- #
DIABETES = DatasetSpec(
    name="diabetes_130",
    modality="tabular",
    context="health",
    path=str(DATA / "diabetes_130.csv"),
    label_column="readmit_early",          # 1 = readmitted within 30 days
    positive_label=1,
    sensitive_columns=("race", "gender"),
    # A clinical encounter cannot plausibly fall outside these ranges. A value
    # outside them is a data-integrity fault with direct physical consequences
    # (it would drive a dosing or triage decision).
    value_ranges={
        "time_in_hospital":     (1, 14),
        "num_lab_procedures":   (0, 132),
        "num_procedures":       (0, 6),
        "num_medications":      (1, 81),
        "number_outpatient":    (0, 42),
        "number_emergency":     (0, 76),
        "number_inpatient":     (0, 21),
        "number_diagnoses":     (1, 16),
    },
    # Strata where an error has the highest physical cost. Each must carry at
    # least `min_stratum_n` records or the dataset under-covers it.
    edge_case_strata={
        "elderly_long_stay":      "age_years >= 80 and time_in_hospital >= 10",
        "emergency_readmitters":  "number_emergency >= 3",
        "high_medication_burden": "num_medications >= 40",
        "many_diagnoses":         "number_diagnoses >= 12",
        "frequent_inpatient":     "number_inpatient >= 5",
    },
    min_stratum_n=500,
    notes="UCI id=296. Stored with readable columns; encoding happens at model time.",
)

# --------------------------------------------------------------------------- #
# 2. Framingham Heart Study  —  context: health, modality: tabular
# --------------------------------------------------------------------------- #
FRAMINGHAM = DatasetSpec(
    name="framingham",
    modality="tabular",
    context="health",
    path=str(DATA / "framingham.csv"),
    label_column="TenYearCHD",             # 1 = coronary heart disease within 10y
    positive_label=1,
    sensitive_columns=("male", "age"),
    value_ranges={
        "age":       (30, 100),
        "totChol":   (80, 700),            # mg/dL
        "sysBP":     (70, 300),            # mmHg
        "diaBP":     (40, 200),            # mmHg
        "BMI":       (12, 70),
        "heartRate": (30, 220),            # bpm
        "glucose":   (30, 500),            # mg/dL
        "cigsPerDay": (0, 100),
    },
    edge_case_strata={
        "hypertensive_crisis": "sysBP >= 180",
        "severe_hyperglycemia": "glucose >= 200",
        "very_high_cholesterol": "totChol >= 300",
        "young_high_risk":      "age <= 40 and prevalentHyp == 1",
        "heavy_smokers":        "cigsPerDay >= 30",
    },
    min_stratum_n=120,
    notes="Teaching version of the Framingham cohort. Add a reference_distribution "
          "for `sex` if you want representation scored against a target population.",
)

# --------------------------------------------------------------------------- #
# 3. Statlog German Credit  —  context: loan_finance, modality: tabular
# --------------------------------------------------------------------------- #
GERMAN_CREDIT = DatasetSpec(
    name="german_credit",
    modality="tabular",
    context="loan_finance",
    path=str(DATA / "german_credit.csv"),
    label_column="bad_credit",             # 1 = bad credit risk
    positive_label=1,
    sensitive_columns=("sex", "age_group"),
    # loan_finance is NOT in the context set for measurement_range_violation or
    # outcome_severity_exposure — those are physical-harm measures and this
    # dataset has no physical outcome. Leaving value_ranges empty makes the
    # N/A reason doubly explicit in the applicability trace.
    value_ranges={},
    edge_case_strata={
        "thin_file_young":     "age_years <= 25 and existing_credits <= 1",
        "large_long_loans":    "credit_amount >= 10000 and duration_months >= 36",
        "unemployed":          "employment_since == 'A71'",
        "foreign_applicants":  "foreign_worker == 'A201'",
    },
    min_stratum_n=40,
    notes="UCI id=144. Small dataset (1000 rows) — min_stratum_n scaled accordingly.",
)

# --------------------------------------------------------------------------- #
# 4. CivilComments  —  context: general, modality: text
# --------------------------------------------------------------------------- #
CIVILCOMMENTS = DatasetSpec(
    name="civilcomments",
    modality="text",
    context="general",
    path=str(DATA / "civilcomments_base.csv"),
    text_column="text",
    clean_pool=str(DATA / "civilcomments_clean.jsonl"),
    toxic_pool=str(DATA / "civilcomments_toxic.jsonl"),
    threat_pool=str(DATA / "civilcomments_threat.jsonl"),
    mix_size=10_000,
    notes="Pools split by the HUMAN annotation, so calibration is not circular.",
)

# --------------------------------------------------------------------------- #
ALL_SPECS = (DIABETES, FRAMINGHAM, GERMAN_CREDIT, CIVILCOMMENTS)
TABULAR_SPECS = tuple(s for s in ALL_SPECS if s.modality == "tabular")
TEXT_SPECS = tuple(s for s in ALL_SPECS if s.modality == "text")
BY_NAME = {s.name: s for s in ALL_SPECS}


def spec_frame():
    import pandas as pd
    return pd.DataFrame([{
        "dataset": s.name, "modality": s.modality, "context": s.context,
        "label_column": s.label_column, "text_column": s.text_column,
        "sensitive_columns": "|".join(s.sensitive_columns),
        "n_value_ranges": len(s.value_ranges),
        "n_edge_case_strata": len(s.edge_case_strata),
        "min_stratum_n": s.min_stratum_n,
        "path": s.path,
    } for s in ALL_SPECS])
