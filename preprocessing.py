from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

df_model = pd.read_csv('koi_clean.csv')
target = 'koi_disposition'
# Separate candidates from training data
train_df = df_model[df_model['koi_disposition'] != 'CANDIDATE'].copy()
candidate_df = df_model[df_model['koi_disposition'] == 'CANDIDATE'].copy()

X_train_raw = train_df.drop(columns=[target])
y_train = train_df[target]

X_candidate = candidate_df.drop(columns=[target])

# Median imputation (fit only on training data)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_raw)
X_candidate_imputed = imputer.transform(X_candidate)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y_train)
# CONFIRMED=0, FALSE POSITIVE=1 (check with le.classes_)
print(le.classes_)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_candidate_scaled = scaler.transform(X_candidate_imputed)

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X_train_scaled, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42)

X_train, X_val, y_train_final, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
# 0.1765 of 85% ≈ 15% of total

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_final)

print(pd.Series(y_train_balanced).value_counts())
import numpy as np

np.save('X_train.npy', X_train_balanced)
np.save('y_train.npy', y_train_balanced)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('X_candidates.npy', X_candidate_scaled)

print("All splits saved.")