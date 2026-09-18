"""
train_model.py
Trains a RandomForest to classify FPR5 vs FPR12 from extracted features,
using a session-grouped split so clips from the same recording session
never appear in both train and test.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("features.csv")

feature_cols = [c for c in df.columns if c not in ("label", "file", "session")]
X = df[feature_cols].values
y = df["label"].values
groups = df["session"].values

# Sanity check: need at least 2 sessions per class for a group split to work
session_counts = df.groupby("label")["session"].nunique()
print("Sessions per class:")
print(session_counts)
if (session_counts < 2).any():
    print("\nWARNING: at least one class has only 1 session. "
          "The group split cannot hold out a session for that class — "
          "results will not reflect real generalization. Record more sessions.")

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Group-aware split: entire sessions go to either train or test, never both
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_enc[train_idx], y_enc[test_idx]
groups_train = groups[train_idx]

print(f"\nTrain: {len(X_train)} clips | Test: {len(X_test)} clips")
print(f"Test classes present: {le.inverse_transform(np.unique(y_test))}")

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Group-aware cross-validation on the training set
n_train_groups = len(set(groups_train))
if n_train_groups >= 2:
    cv = GroupKFold(n_splits=min(5, n_train_groups))
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, groups=groups_train)
    print(f"\nGrouped CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
else:
    print("\nSkipping CV: not enough distinct sessions in training set.")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nTest set performance:")
print(classification_report(y_test, y_pred,
                            labels=np.unique(np.concatenate([y_test, y_pred])),
                            target_names=le.inverse_transform(
                                np.unique(np.concatenate([y_test, y_pred])))))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=feature_cols)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

joblib.dump(clf, "fpr_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\nSaved model to fpr_classifier.pkl")