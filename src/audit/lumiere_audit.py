"""
LUMIERE Dataset Audit — Fase 0
==============================
Esplora i quattro CSV del dataset LUMIERE seguendo le EDA Guidelines
definite in CONTEXT.md:

REGOLA FONDAMENTALE: l'unità di analisi è il PAZIENTE, non lo scan.
Ogni statistica viene calcolata prima per paziente, poi aggregata.

Posizione nel repo: src/audit/lumiere_audit.py
Esecuzione: python -m src.audit.lumiere_audit (dalla root del progetto)
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/lumiere")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Classi RANO da escludere (non valutazioni cliniche standard)
RANO_EXCLUDE = {"Pre-Op", "Post-Op", "Post-Op ", "Post-Op/PD"}

# Mapping classi RANO → 3 classi
RANO_MAPPING = {
    "CR": "Response",
    "PR": "Response",
    "SD": "Stable",
    "PD": "Progressive",
}


# ---------------------------------------------------------------------------
# Step 1 — Audit grezzo: struttura di ogni CSV
# ---------------------------------------------------------------------------
def audit_raw_files() -> None:
    """Stampa struttura, shape, colonne e missing values di ogni CSV."""
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"❌ Nessun CSV trovato in {DATA_DIR}")
        print("Copia i file in data/raw/lumiere/ prima di procedere.")
        return

    print(f"📂 File trovati: {[f.name for f in csv_files]}\n")

    for filepath in sorted(csv_files):
        print(f"\n{'='*60}")
        print(f"📄 {filepath.name}")
        print(f"{'='*60}")

        df = pd.read_csv(filepath)
        print(f"Shape: {df.shape[0]} righe × {df.shape[1]} colonne")
        print(f"\nColonne:\n{df.columns.tolist()}")
        print(f"\nPrime 3 righe:\n{df.head(3).to_string()}")

        # Missing values
        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        nulli = null_pct[null_pct > 0]
        if not nulli.empty:
            print(f"\n⚠️  Missing values (% per colonna):\n{nulli.to_string()}")
        else:
            print("\n✅ Nessun missing value")


# ---------------------------------------------------------------------------
# Step 2 — Audit RANO: distribuzione classi PER PAZIENTE
# ---------------------------------------------------------------------------
def audit_rano() -> pd.DataFrame:
    """
    Analizza le label RANO seguendo la regola per-paziente.
    Restituisce il DataFrame filtrato e mappato.
    """
    print(f"\n{'='*60}")
    print("📊 AUDIT RANO — per paziente")
    print(f"{'='*60}")

    rano = pd.read_csv(DATA_DIR / "LUMIERE-ExpertRating-v202211.csv")
    rano.columns = ["Patient", "Date", "LessThan3M", "NonMeasurable", "Rating", "Rationale"]

    print(f"Timepoint totali nel file: {len(rano)}")
    print(f"Distribuzione Rating (raw):\n{rano['Rating'].value_counts().to_string()}")

    # Filtra timepoint non validi
    rano_valid = rano[~rano["Rating"].isin(RANO_EXCLUDE)].copy()
    rano_valid["Rating_grouped"] = rano_valid["Rating"].map(RANO_MAPPING)

    print(f"\n✅ Timepoint validi dopo esclusione Pre/Post-Op: {len(rano_valid)}")
    print(f"✅ Pazienti con almeno 1 timepoint valido: {rano_valid['Patient'].nunique()}")

    # --- PER PAZIENTE: distribuzione timepoint ---
    tp_per_patient = rano_valid.groupby("Patient")["Date"].count()
    print(f"\n📊 Timepoint validi per paziente (statistica per paziente):")
    print(tp_per_patient.describe().round(1).to_string())

    for threshold in [2, 3, 4, 5]:
        n = (tp_per_patient >= threshold).sum()
        print(f"  Pazienti con >= {threshold} timepoint: {n}")

    # --- PER PAZIENTE: distribuzione classi ---
    # Conta quanti pazienti hanno almeno 1 occorrenza di ogni classe
    print(f"\n📊 Distribuzione classi RANO (per timepoint — grezzo):")
    print(rano_valid["Rating_grouped"].value_counts().to_string())

    print(f"\n📊 Distribuzione classi RANO (per paziente — quanti pazienti hanno almeno 1):")
    for classe in ["Progressive", "Stable", "Response"]:
        pazienti_con_classe = rano_valid[rano_valid["Rating_grouped"] == classe]["Patient"].nunique()
        print(f"  {classe}: {pazienti_con_classe} pazienti")

    # Pazienti dominanti (alto numero di scan)
    print(f"\n⚠️  Top 5 pazienti per numero di timepoint (rischio dominanza):")
    print(tp_per_patient.sort_values(ascending=False).head(5).to_string())

    return rano_valid


# ---------------------------------------------------------------------------
# Utility — parse settimane dal formato "week-044" o "week-000-1"
# ---------------------------------------------------------------------------
def parse_week(date_str: str) -> float:
    """
    Converte stringhe tipo 'week-044' o 'week-000-1' in un numero float
    ordinale di settimane. Il suffisso '-1', '-2' indica scan multipli
    nella stessa settimana — viene trattato come offset di 0.1, 0.2 ecc.
    Esempi:
        'week-044'   → 44.0
        'week-000-1' → 0.1
        'week-000-2' → 0.2
    """
    import re
    parts = date_str.replace("week-", "").split("-")
    week = float(parts[0])
    if len(parts) > 1:
        week += float(parts[1]) * 0.1
    return week


# ---------------------------------------------------------------------------
# Step 3 — Audit intervalli temporali (clinical workflow leakage check)
# ---------------------------------------------------------------------------
def audit_temporal_intervals(rano_valid: pd.DataFrame) -> None:
    """
    Analizza gli intervalli Δt tra scan consecutivi per paziente.
    Questo è il primo controllo per il clinical workflow leakage.
    Le date sono nel formato 'week-NNN' — vengono convertite in settimane numeriche.
    """
    print(f"\n{'='*60}")
    print("⏱️  AUDIT INTERVALLI TEMPORALI — clinical workflow leakage check")
    print(f"{'='*60}")

    rano_valid = rano_valid.copy()
    rano_valid["week_num"] = rano_valid["Date"].apply(parse_week)
    rano_valid = rano_valid.sort_values(["Patient", "week_num"])

    # Calcola Δt in settimane per ogni coppia consecutiva
    delta_records = []
    for patient, group in rano_valid.groupby("Patient"):
        weeks = group["week_num"].values
        ratings = group["Rating_grouped"].values
        for i in range(len(weeks) - 1):
            delta_weeks = weeks[i + 1] - weeks[i]
            delta_records.append({
                "patient": patient,
                "delta_weeks": delta_weeks,
                "rating_t": ratings[i],
                "rating_t1": ratings[i + 1],
            })

    delta_df = pd.DataFrame(delta_records)

    print(f"\n📊 Distribuzione Δt in settimane (per coppia consecutiva):")
    print(delta_df["delta_weeks"].describe().round(1).to_string())

    print(f"\n📊 Δt medio per classe RANO al timepoint T+1 (segnale leakage):")
    print(delta_df.groupby("rating_t1")["delta_weeks"].mean().round(1).to_string())
    print(
        "\n⚠️  Se Progressive ha Δt significativamente più basso degli altri → "
        "clinical workflow leakage attivo. Da dichiarare nel paper."
    )


# ---------------------------------------------------------------------------
# Step 4 — Audit feature radiomiche: missing values e completezza
# ---------------------------------------------------------------------------
def audit_radiomic_features() -> None:
    """
    Analizza il CSV delle feature PyRadiomics:
    completezza per scan e distribuzione missing per feature.
    """
    print(f"\n{'='*60}")
    print("🔬 AUDIT FEATURE RADIOMICHE")
    print(f"{'='*60}")

    feat = pd.read_csv(DATA_DIR / "LUMIERE-pyradiomics-hdglioauto-features.csv")
    print(f"Shape: {feat.shape}")
    print(f"\nPrime colonne: {feat.columns[:10].tolist()}")

    # Missing values per feature
    null_pct = (feat.isnull().sum() / len(feat) * 100).round(1)
    nulli = null_pct[null_pct > 0]
    print(f"\n⚠️  Feature con missing values: {len(nulli)}")
    if not nulli.empty:
        print(nulli.sort_values(ascending=False).head(20).to_string())

    # Distribuzione valori per feature (skewness)
    numeric_cols = feat.select_dtypes(include="number").columns
    skewness = feat[numeric_cols].skew().abs()
    high_skew = skewness[skewness > 2]
    print(f"\n⚠️  Feature con skewness > 2 (candidate a log-transform): {len(high_skew)}")
    if not high_skew.empty:
        print(high_skew.sort_values(ascending=False).head(10).to_string())


# ---------------------------------------------------------------------------
# Step 5 — Calcolo n_effettiva (paired examples dopo label shift)
# ---------------------------------------------------------------------------
def compute_n_effettiva(rano_valid: pd.DataFrame) -> dict:
    """
    Calcola quanti esempi paired (features_t, label_t+1) esistono
    dopo il label shift. Questa è la vera sample size del progetto.
    """
    print(f"\n{'='*60}")
    print("🎯 N_EFFETTIVA — paired examples dopo label shift")
    print(f"{'='*60}")

    rano_valid = rano_valid.copy()
    rano_valid["week_num"] = rano_valid["Date"].apply(parse_week)
    rano_valid = rano_valid.sort_values(["Patient", "week_num"])

    paired = []
    for patient, group in rano_valid.groupby("Patient"):
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):  # escludi sempre l'ultimo
            paired.append({
                "patient": patient,
                "week_t": rows.loc[i, "week_num"],
                "week_t1": rows.loc[i + 1, "week_num"],
                "delta_weeks": rows.loc[i + 1, "week_num"] - rows.loc[i, "week_num"],
                "label_t1": rows.loc[i + 1, "Rating_grouped"],
            })

    paired_df = pd.DataFrame(paired)
    n_effettiva = len(paired_df)
    n_pazienti = paired_df["patient"].nunique()

    print(f"\n✅ n_effettiva (paired examples totali): {n_effettiva}")
    print(f"✅ Pazienti rappresentati: {n_pazienti}")
    print(f"\n📊 Distribuzione per classe target (label_t+1):")
    print(paired_df["label_t1"].value_counts().to_string())

    stats = {
        "n_effettiva": n_effettiva,
        "n_pazienti": n_pazienti,
        "distribuzione_classi": paired_df["label_t1"].value_counts().to_dict(),
    }

    # Salva stats in JSON per riferimento futuro
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Stats salvate in {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 GBM Longitudinal Toolkit — LUMIERE Audit")
    print("=" * 60)

    audit_raw_files()
    rano_valid = audit_rano()
    audit_temporal_intervals(rano_valid)
    audit_radiomic_features()
    stats = compute_n_effettiva(rano_valid)

    print(f"\n{'='*60}")
    print("✅ AUDIT COMPLETATO")
    print(f"   n_effettiva = {stats['n_effettiva']} paired examples")
    print(f"   Pazienti = {stats['n_pazienti']}")
    print("=" * 60)