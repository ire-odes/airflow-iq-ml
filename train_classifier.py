"""
train_classifier.py
Binary FPR5/FPR12 RandomForest, following the paper's protocol:

  1. 25% pure holdout / 75% training pool  (their Section 5.3)
     -- but split by SESSION GROUP, which they didn't do and you must.
  2. Grid search with 5-fold cross-validation on the training pool,
     folds also grouped by session.
  3. Grid informed by their sensitivity findings (Figs. 13-16):
     shallow trees won, more trees monotonically helped, short segments
     won (your clips are already ~1-2 s, so that knob is fixed).
  4. Zero-importance feature pruning (their Section 5.4), retrain,
     verify no accuracy loss.
  5. Report in their Table 7 style: accuracy + FP rate (clean called
     dirty) + FN rate (dirty called clean). FN is the expensive error.

Outputs: model_bundle.pkl -- a single dict {classifier, label_encoder,
         feature_cols, selected_features}, one file so the group of four
         is written atomically as a unit (see the persist step below for
         why that matters -- it wasn't always one file).
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit)
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
DIRTY_LABEL = "FPR12"   # positive class for FP/FN accounting


def load_data(path="features.csv"):
    df = pd.read_csv(path)
    meta_cols = ["label", "file", "session"]
    LOUDNESS_COLS = ["mfcc0_mean", "mfcc0_std", "rms_mean", "rms_std"]
    feature_cols = [c for c in df.columns
                    if c not in meta_cols and c not in LOUDNESS_COLS]
    X = df[feature_cols].values
    groups = df["session"].values

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    return X, y, groups, feature_cols, le


def fp_fn_rates(y_true, y_pred, dirty_idx):
    """Paper Table 7 convention, adapted to binary:
    FP = clean filter reported dirty (cheap error, they *preferred* it)
    FN = dirty filter reported clean (the expensive one)"""
    clean_mask = y_true != dirty_idx
    dirty_mask = y_true == dirty_idx
    fp = np.mean(y_pred[clean_mask] == dirty_idx) if clean_mask.any() else 0.0
    fn = np.mean(y_pred[dirty_mask] != dirty_idx) if dirty_mask.any() else 0.0
    return fp, fn


def main():
    X, y, groups, feature_cols, le = load_data()
    dirty_idx = int(np.where(le.classes_ == DIRTY_LABEL)[0][0])
    print(f"{len(X)} clips, {len(feature_cols)} features, "
          f"{len(set(groups))} sessions")

    # ---- 1. Session-grouped 75/25 holdout (paper split, leak-free) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = groups[train_idx]

    print(f"Train: {len(X_tr)} clips / {len(set(g_tr))} sessions | "
          f"Holdout: {len(X_te)} clips / {len(set(groups[test_idx]))} sessions")

    # ---- 2. Grid search, 5-fold grouped CV (paper Section 5.3) --------
    # Grid shaped by their sensitivity curves:
    #   Fig. 13: shallow trees (5 splits) beat deep -> small max_leaf_nodes
    #   Fig. 15: accuracy still climbing at 1000 trees, diminishing returns
    n_cv = min(5, len(set(g_tr)))
    if n_cv < 5:
        print(f"WARNING: only {n_cv} sessions in training pool; "
              f"using {n_cv}-fold grouped CV instead of 5.")
    param_grid = {
        "n_estimators": [100, 300, 1000],
        "max_leaf_nodes": [8, 16, 32],      # analog of their max-splits axis
        "min_samples_leaf": [1, 5],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE,
                                class_weight="balanced", n_jobs=-1)
    search = GridSearchCV(rf, param_grid, cv=GroupKFold(n_splits=n_cv),
                          scoring="accuracy", n_jobs=-1)
    search.fit(X_tr, y_tr, groups=g_tr)

    print(f"\nBest params: {search.best_params_}")
    print(f"Insample (grouped 5-fold CV) accuracy: "
          f"{search.best_score_:.2%}")

    # ---- 3. Holdout evaluation, full feature set ----------------------
    model = search.best_estimator_
    y_pred = model.predict(X_te)
    acc_full = accuracy_score(y_te, y_pred)
    fp, fn = fp_fn_rates(y_te, y_pred, dirty_idx)

    print(f"\n== Holdout, full feature set ==")
    print(f"Outsample accuracy: {acc_full:.2%}")
    print(f"False positives (clean->dirty): {fp:.2%}   <- cheap error")
    print(f"False negatives (dirty->clean): {fn:.2%}   <- expensive error")
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # ---- 4. Zero-importance pruning (paper Section 5.4) ---------------
    importances = model.feature_importances_
    keep_mask = importances > 0
    kept = [f for f, k in zip(feature_cols, keep_mask) if k]
    print(f"Pruning: {len(feature_cols)} -> {len(kept)} features "
          f"({1 - len(kept)/len(feature_cols):.1%} eliminated)")

    model_pruned = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced",
        n_jobs=-1, **search.best_params_)
    model_pruned.fit(X_tr[:, keep_mask], y_tr)
    y_pred_p = model_pruned.predict(X_te[:, keep_mask])
    acc_pruned = accuracy_score(y_te, y_pred_p)
    print(f"Holdout accuracy after pruning: {acc_pruned:.2%} "
          f"(delta {acc_pruned - acc_full:+.2%})")

    # Their claim: pruning should cost nothing, since the forest already
    # ignored zero-importance features. If it *does* drop, keep the full
    # set -- the pruned set is only for on-device payload savings.
    final_model = model_pruned if acc_pruned >= acc_full - 0.01 else model
    final_features = kept if final_model is model_pruned else feature_cols

    top = sorted(zip(feature_cols, importances),
                 key=lambda t: -t[1])[:10]
    print("\nTop 10 features:")
    for name, imp in top:
        print(f"  {name:28s} {imp:.4f}")

    # ---- 5. Persist, atomically -----------------------------------------
    # The classifier, label encoder, feature order, and selected-feature
    # list must always agree with each other -- every consumer loads them
    # assuming they came from the same run. Writing them as 4 separate
    # files, even with each individual write made atomic (temp file +
    # rename), still isn't atomic as a GROUP: a process killed between two
    # of the four renames leaves some files updated and others stale --
    # which happened in practice, three times, always the same failure
    # mode (a fresh classifier paired with a stale selected_features.pkl,
    # silently breaking every downstream consumer with a feature-count
    # mismatch deep inside sklearn). A single bundle file means a single
    # rename, which genuinely is atomic (same-directory os.replace is
    # atomic on both POSIX and Windows) -- there is no partial-write state
    # for a group of one.
    bundle = {
        "classifier": final_model,
        "label_encoder": le,
        "feature_cols": feature_cols,        # extraction order
        "selected_features": final_features,  # what model eats
    }
    tmp_path = "model_bundle.pkl.tmp"
    joblib.dump(bundle, tmp_path)
    os.replace(tmp_path, "model_bundle.pkl")
    print("\nSaved model_bundle.pkl (classifier, label_encoder, feature_cols, selected_features)")


if __name__ == "__main__":
    main()
