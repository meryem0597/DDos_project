# -*- coding: utf-8 -*-
"""
Détection et prédiction d'attaques DDoS — application Streamlit.

La logique de traitement reprend celle du notebook « projet ddos attaque » :
doublons -> fillna(0) -> exclusion des identifiants -> get_dummies ->
split 70/15/15 stratifié -> StandardScaler (fit sur train) -> 5 modèles.
Le jeu de données est SYNTHÉTIQUE (usage pédagogique).
"""
import hashlib
import hmac
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent / "dataset.csv"
DROP_COLS = ["label", "timestamp", "src_ip", "dst_ip", "src_port"]
NUM_FEATURES = [
    "duration_ms", "packet_count", "byte_count", "packets_per_second",
    "bytes_per_second", "avg_packet_size_bytes", "syn_count", "ack_count",
    "connections_10s", "unique_sources_10s",
]
SCALED_MODELS = {"Régression linéaire", "Régression logistique", "SVM"}

# ---------------------------------------------------------------- Authentification
# Seuls les empreintes SHA-256 sont stockées dans le code (pas les mots de passe en clair).
USER_HASH = hashlib.sha256(b"meryem").hexdigest()
PASS_HASH = hashlib.sha256(b"errachidy").hexdigest()


def _check_credentials(user: str, pwd: str) -> bool:
    u = hashlib.sha256(user.strip().encode()).hexdigest()
    p = hashlib.sha256(pwd.encode()).hexdigest()
    return hmac.compare_digest(u, USER_HASH) & hmac.compare_digest(p, PASS_HASH)


def login_page() -> None:
    st.title("🛡️ Détection d'attaques DDoS")
    st.caption("Veuillez vous authentifier pour accéder à l'application.")
    _, col, _ = st.columns([1, 2, 1])
    with col, st.form("login"):
        user = st.text_input("Login")
        pwd = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter", type="primary"):
            if _check_credentials(user, pwd):
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Login ou mot de passe incorrect.")


# ---------------------------------------------------------------- Données & modèles
@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy().fillna(0)  # même ordre que le notebook


def encode(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return pd.get_dummies(X, columns=["protocol", "dst_port"], dtype=int)


@st.cache_resource(show_spinner="Entraînement des 5 modèles…")
def train_all() -> dict:
    df = clean(load_raw())
    y = df["label"]
    X = encode(df)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val, Xs_test = scaler.transform(X_val), scaler.transform(X_test)

    models = {
        "Régression linéaire": LinearRegression().fit(Xs_train, y_train),
        "Régression logistique": LogisticRegression(
            max_iter=1000, random_state=42).fit(Xs_train, y_train),
        "SVM": SVC(kernel="rbf", C=1.0).fit(Xs_train, y_train),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=2).fit(X_train, y_train),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=2).fit(X_train, y_train),
    }

    def predict(name, X_raw, X_scaled):
        data = X_scaled if name in SCALED_MODELS else X_raw
        out = models[name].predict(data)
        return (out >= 0.5).astype(int) if name == "Régression linéaire" else out

    def metrics(y_true, y_pred) -> dict:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Précision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-score": f1_score(y_true, y_pred, zero_division=0),
        }

    rows = [{"Modèle": n, **metrics(y_val, predict(n, X_val, Xs_val))} for n in models]
    table = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best = table.iloc[0]["Modèle"]
    y_pred_test = predict(best, X_test, Xs_test)

    return {
        "models": models, "scaler": scaler, "columns": X.columns.tolist(),
        "table": table, "best": best,
        "test_metrics": metrics(y_test, y_pred_test),
        "y_test": y_test, "y_pred_test": y_pred_test,
        "sizes": (len(X_train), len(X_val), len(X_test)),
        "clean_df": df,
    }


def predict_frame(art: dict, model_name: str, raw: pd.DataFrame):
    """Retourne (classes 0/1, score DDoS) pour un DataFrame au format du CSV."""
    X = encode(raw.fillna(0)).reindex(columns=art["columns"], fill_value=0)
    model = art["models"][model_name]
    data = art["scaler"].transform(X) if model_name in SCALED_MODELS else X
    if model_name == "Régression linéaire":
        score = np.clip(model.predict(data), 0, 1)
        return (score >= 0.5).astype(int), score
    if model_name == "SVM":  # pas de probabilité native : sigmoïde de la distance à la frontière
        return model.predict(data), 1 / (1 + np.exp(-model.decision_function(data)))
    return model.predict(data), model.predict_proba(data)[:, 1]


# ---------------------------------------------------------------- Pages
def page_stats(raw: pd.DataFrame, art: dict) -> None:
    df = art["clean_df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes (brutes)", f"{len(raw):,}")
    c2.metric("Colonnes", raw.shape[1])
    c3.metric("Doublons supprimés", int(raw.duplicated().sum()))
    c4.metric("Valeurs manquantes (NaN)", int(raw.drop_duplicates().isna().sum().sum()))

    st.subheader("Aperçu des données")
    st.dataframe(raw.head(10))

    left, right = st.columns(2)
    with left:
        st.subheader("Répartition des classes")
        counts = df["label"].map({0: "Normal", 1: "DDoS"}).value_counts()
        st.bar_chart(counts)
        st.caption(" | ".join(f"{k} : {v} ({v / len(df):.1%})" for k, v in counts.items()))
    with right:
        st.subheader("Valeurs manquantes par colonne")
        na = raw.isna().sum()
        st.dataframe(na[na > 0].rename("NaN").to_frame())
        st.caption("Traitement appliqué : `fillna(0)` après suppression des doublons (comme le notebook).")

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().T.round(2))

    st.subheader("Moyennes par classe")
    st.dataframe(df.groupby(df["label"].map({0: "Normal", 1: "DDoS"}))[NUM_FEATURES].mean().T.round(2))


def page_metrics(art: dict) -> None:
    n_tr, n_va, n_te = art["sizes"]
    st.caption(f"Découpage stratifié : Train {n_tr} · Validation {n_va} · Test {n_te} (70 / 15 / 15 %).")

    st.subheader("Comparaison des modèles (validation)")
    st.dataframe(art["table"].style.format({c: "{:.3f}" for c in art["table"].columns[1:]})
                 .highlight_max(subset=art["table"].columns[1:], color="#1b5e20"))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(art["table"]["Modèle"], art["table"]["F1-score"], color="#1f77b4")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score (validation)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Évaluation finale sur le test — {art['best']}")
    m = art["test_metrics"]
    cols = st.columns(4)
    for col, (name, val) in zip(cols, m.items()):
        col.metric(name, f"{val:.3f}")

    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        art["y_test"], art["y_pred_test"], display_labels=["Normal", "DDoS"],
        cmap="Blues", ax=ax2)
    ax2.set_title("Matrice de confusion — test")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.info("Faux négatifs = attaques non détectées ; faux positifs = trafic normal signalé à tort. "
            "Les données étant synthétiques, ces scores ne prouvent pas une efficacité en production.")


def _show_verdict(label: int, score: float, model_name: str) -> None:
    if label == 1:
        st.error(f"🚨 **DDoS détecté** — score d'attaque : {score:.1%}")
    else:
        st.success(f"✅ **Trafic Normal** — score d'attaque : {score:.1%}")
    if model_name in ("Régression linéaire", "SVM"):
        st.caption(f"Pour « {model_name} », le score n'est pas une vraie probabilité (indicateur de confiance).")


def page_predict(art: dict) -> None:
    df = art["clean_df"]
    model_name = st.selectbox(
        "Modèle", list(art["models"]), index=list(art["models"]).index(art["best"]),
        help="Par défaut : le meilleur modèle selon le F1-score de validation.")
    tab_one, tab_batch = st.tabs(["Flux unique", "Fichier CSV (lot)"])

    # ---- flux unique
    with tab_one:
        if "seed" not in st.session_state:
            st.session_state["seed"] = 0
        c1, c2 = st.columns([3, 1])
        preset = c1.radio("Pré-remplir avec", ["Valeurs médianes", "Exemple Normal", "Exemple DDoS"],
                          horizontal=True)
        if c2.button("🎲 Autre exemple"):
            st.session_state["seed"] += 1

        if preset == "Valeurs médianes":
            base = df[NUM_FEATURES].median()
            proto0, port0 = df["protocol"].mode()[0], int(df["dst_port"].mode()[0])
        else:
            subset = df[df["label"] == (1 if preset == "Exemple DDoS" else 0)]
            row = subset.sample(1, random_state=st.session_state["seed"]).iloc[0]
            base, proto0, port0 = row[NUM_FEATURES], row["protocol"], int(row["dst_port"])
        k = f"{preset}_{st.session_state['seed']}"

        protos = sorted(df["protocol"].unique())
        ports = sorted(int(p) for p in df["dst_port"].unique())
        a, b = st.columns(2)
        protocol = a.selectbox("Protocole", protos, index=protos.index(proto0), key=f"p{k}")
        dst_port = b.selectbox("Port de destination", ports, index=ports.index(port0), key=f"d{k}")

        values = {}
        cols = st.columns(3)
        for i, feat in enumerate(NUM_FEATURES):
            values[feat] = cols[i % 3].number_input(
                feat, min_value=0.0, value=float(base[feat]), key=f"{feat}{k}")

        if st.button("Prédire", type="primary"):
            sample = pd.DataFrame([{"protocol": protocol, "dst_port": dst_port, **values}])
            label, score = predict_frame(art, model_name, sample)
            _show_verdict(int(label[0]), float(score[0]), model_name)

    # ---- lot
    with tab_batch:
        st.write("Colonnes requises : `protocol`, `dst_port`, " + ", ".join(f"`{c}`" for c in NUM_FEATURES) + ".")
        up = st.file_uploader("Charger un CSV de flux", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            missing = [c for c in ["protocol", "dst_port", *NUM_FEATURES] if c not in new.columns]
            if missing:
                st.error(f"Colonnes manquantes : {', '.join(missing)}")
            else:
                label, score = predict_frame(art, model_name, new)
                out = new.copy()
                out["prediction"] = np.where(label == 1, "DDoS", "Normal")
                out["score_ddos"] = score.round(4)
                n_ddos = int((label == 1).sum())
                st.metric("Flux classés DDoS", f"{n_ddos} / {len(out)}")
                st.dataframe(out)
                st.download_button("Télécharger les résultats", out.to_csv(index=False).encode(),
                                   "predictions_ddos.csv", "text/csv")


# ---------------------------------------------------------------- Main
def main() -> None:
    st.set_page_config(page_title="Détection DDoS", page_icon="🛡️", layout="wide")
    if not st.session_state.get("auth"):
        login_page()
        return

    with st.sidebar:
        st.write("👤 Connecté : **meryem**")
        if st.button("Se déconnecter"):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.caption("Jeu de données synthétique à visée pédagogique.")

    st.title("🛡️ Détection et prédiction d'attaques DDoS")
    if not DATA_PATH.exists():
        st.error("dataset.csv introuvable : placez-le dans le même dossier que app.py.")
        return

    raw, art = load_raw(), train_all()
    t1, t2, t3 = st.tabs(["📊 Statistiques", "📈 Métriques des modèles", "🔮 Prédiction"])
    with t1:
        page_stats(raw, art)
    with t2:
        page_metrics(art)
    with t3:
        page_predict(art)


main()
