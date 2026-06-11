import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1 — Load Data
# ---------------------------------------------------------
print("Loading data...")
# Note: adjust to pd.read_csv if files are saved as CSV despite the .xlsx name
df1 = pd.read_csv("case_study1new.csv") 
df2 = pd.read_csv("case_study2new.csv")

# ---------------------------------------------------------
# Step 2 — Handle -99999 Sentinel Values
# ---------------------------------------------------------
print("Cleaning -99999 values...")
# Drop columns from df2 where count of -99999 > 10,000
cols_to_drop = [col for col in df2.columns if (df2[col] == -99999).sum() > 10000]
df2.drop(columns=cols_to_drop, inplace=True)

# Remove rows from both df1 and df2 where any column = -99999
df1 = df1[(df1 != -99999).all(axis=1)]
df2 = df2[(df2 != -99999).all(axis=1)]

# ---------------------------------------------------------
# Step 3 — Merge
# ---------------------------------------------------------
print("Merging datasets...")
df = pd.merge(df1, df2, on='PROSPECTID', how='inner')
print(f"Merged dataset shape: {df.shape}")

# ---------------------------------------------------------
# Step 4 — EDA 
# ---------------------------------------------------------
print("Generating EDA charts...")
# Bar chart: Approved_Flag distribution
plt.figure()
sns.countplot(data=df, x='Approved_Flag')
plt.title('Approved Flag Distribution')
plt.savefig('eda_approved_flag.png')

# Box/violin: Age, Income (NETMONTHLYINCOME), Credit_Score (if applicable/available) by Approved_Flag
for col in ['AGE', 'NETMONTHLYINCOME']:
    if col in df.columns:
        plt.figure()
        sns.boxplot(data=df, x='Approved_Flag', y=col)
        plt.title(f'{col} by Approved_Flag')
        plt.savefig(f'eda_{col}_box.png')

# ---------------------------------------------------------
# Step 5 — Feature Selection
# ---------------------------------------------------------
print("Performing feature selection...")
# Separate Target
y = df['Approved_Flag']
X_temp = df.drop(['PROSPECTID', 'Approved_Flag'], axis=1)

categorical_cols = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
numeric_cols = [col for col in X_temp.columns if col not in categorical_cols]

# 1. Chi-Square for categorical
kept_cat_cols = []
for col in categorical_cols:
    if col in df.columns:
        contingency_table = pd.crosstab(df[col], df['Approved_Flag'])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        if p < 0.05:
            kept_cat_cols.append(col)

# 2. ANOVA for numeric features
kept_num_cols = []
for col in numeric_cols:
    groups = [df[df['Approved_Flag'] == flag][col].dropna() for flag in df['Approved_Flag'].unique()]
    stat, p = f_oneway(*groups)
    if p < 0.05:
        kept_num_cols.append(col)

# 3. VIF (multicollinearity)
X_vif = df[kept_num_cols].copy()

# Iterative VIF drop function
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Drop features with VIF > 6 iteratively
while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df['VIF'].max()
    if max_vif > 6:
        max_vif_feature = vif_df.loc[vif_df['VIF'] == max_vif, 'feature'].values[0]
        X_vif.drop(columns=[max_vif_feature], inplace=True)
    else:
        break

final_num_cols = list(X_vif.columns)
final_features = kept_cat_cols + final_num_cols
df_final = df[final_features + ['Approved_Flag']].copy()

print(f"Features remaining after selection: {len(final_features)}")

# ---------------------------------------------------------
# Step 6 — Encode and Scale
# ---------------------------------------------------------
print("Encoding and scaling...")
le = LabelEncoder()
for col in kept_cat_cols:
    df_final[col] = le.fit_transform(df_final[col])

# Encode target
df_final['Approved_Flag'] = df_final['Approved_Flag'].map({'P1':1, 'P2':2, 'P3':3, 'P4':4})

# Split
X = df_final[final_features]
# Shift y by -1 for XGBoost (expects 0 to 3 instead of 1 to 4)
y = df_final['Approved_Flag'] - 1 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 7 — Train 5 Models
# ---------------------------------------------------------
print("Training models...")
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                             eval_metric='mlogloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=['P1', 'P2', 'P3', 'P4']))

# ---------------------------------------------------------
# Step 8 — Validate Best Model (XGBoost)
# ---------------------------------------------------------
print("Cross-validating XGBoost...")
xgb_model = models["XGBoost"]
cv = cross_val_score(xgb_model, X_train_sc, y_train, 
                     cv=StratifiedKFold(5), scoring='accuracy')
print(f"CV Mean Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# ---------------------------------------------------------
# Step 9 — Feature Importance
# ---------------------------------------------------------
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances (XGBoost)")
plt.bar(range(20), importances[indices], align="center")
plt.xticks(range(20), [final_features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png')

# ---------------------------------------------------------
# Step 10 — Save Model
# ---------------------------------------------------------
print("Saving artifacts...")
pickle.dump(xgb_model, open('credit_risk_model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))
pickle.dump(final_features, open('feature_names.pkl','wb'))
print("Pipeline complete. Model and artifacts saved.")