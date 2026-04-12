import pandas as pd

df = pd.read_csv('cumulative_2026.04.12_06.34.10.csv', comment='#')
print(df.shape)
print(df['koi_disposition'].value_counts())
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Drop columns with >50% missing
threshold = 0.5
df_clean = df.dropna(thresh=len(df) * threshold, axis=1)
print(df_clean.shape)

# Drop non-feature columns
drop_cols = [
    'rowid', 'kepid', 'kepoi_name', 'koi_vet_stat', 'koi_vet_date',
    'koi_pdisposition',
    'koi_disp_prov', 'koi_comment', 'koi_fittype', 'koi_limbdark_mod',
    'koi_parm_prov', 'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod',
    'koi_datalink_dvr', 'koi_datalink_dvs', 'koi_sparprov',
    'koi_eccen'
]

target = 'koi_disposition'
df_model = df_clean.drop(columns=drop_cols)

X = df_model.drop(columns=[target])
y = df_model[target]

print(f"Features: {X.shape}")
print(X.dtypes.value_counts())

# Save cleaned dataset
df_model.to_csv('koi_clean.csv', index=False)
print("Saved koi_clean.csv")