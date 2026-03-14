import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Prices Prediction", page_icon="🏠", layout="wide")

st.title("🏠 House Prices Prediction")
st.write("**Midterm Project** — Upload `train.csv` from Kaggle to run the full pipeline.")

uploaded_file = st.file_uploader("Upload train.csv", type="csv")

if uploaded_file is None:
    st.info("👆 Upload train.csv to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 2: Data Overview ─────────────────────────────────────────────────────
st.header("Step 2: Data Overview")
st.dataframe(df.head())
st.dataframe(df.describe())
st.write("**Data types:**")
st.write(df.dtypes.value_counts())

# ── Step 3: Data Analysis ─────────────────────────────────────────────────────
st.header("Step 3: Data Analysis")

st.subheader("3.1 Looking at SalePrice")
st.write("SalePrice is skewed to the right. Taking the log makes it look more like a bell curve, which helps the models learn better.")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['SalePrice'], bins=50, color='pink', edgecolor='white')
    ax.set_title('SalePrice Distribution')
    ax.set_xlabel('Price')
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log1p(df['SalePrice']), bins=50, color='teal', edgecolor='white')
    ax.set_title('Log(SalePrice) Distribution')
    ax.set_xlabel('log(Price)')
    st.pyplot(fig)

st.write(f"Skewness (original): **{round(df['SalePrice'].skew(), 3)}**")
st.write(f"Skewness (log): **{round(np.log1p(df['SalePrice']).skew(), 3)}**")

st.subheader("3.2 What features are most correlated with SalePrice?")
corr = df.select_dtypes(include=np.number).corr()['SalePrice'].sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    corr[1:11].plot(kind='barh', color='pink', ax=ax)
    ax.set_title('Top 10 Features Correlated with SalePrice')
    ax.set_xlabel('Correlation')
    plt.tight_layout()
    st.pyplot(fig)
with col2:
    top10_cols = corr.head(10).index.tolist() + ['SalePrice']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[top10_cols].corr(), annot=True, fmt='.2f', cmap='RdPu', square=True, ax=ax)
    ax.set_title('Correlation Matrix - Top 10 Features')
    plt.tight_layout()
    st.pyplot(fig)

st.write(corr[1:11])

st.subheader("Top 4 Features vs SalePrice")
top4 = corr[1:5].index.tolist()
cols = st.columns(4)
for i, col in enumerate(top4):
    with cols[i]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(df[col], df['SalePrice'], alpha=0.4, s=10, color='#ADD0B3')
        ax.set_xlabel(col)
        ax.set_ylabel('SalePrice')
        ax.set_title(f'{col} vs SalePrice')
        st.pyplot(fig)

st.subheader("3.3 Removing Outliers")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.4, s=12, color='#ADD0B3')
ax.scatter(
    df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]['GrLivArea'],
    df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]['SalePrice'],
    color='#B43757', s=50, label='Outliers'
)
ax.set_xlabel('GrLivArea')
ax.set_ylabel('SalePrice')
ax.set_title('Spotting Outliers')
ax.legend()
st.pyplot(fig)

df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))].reset_index(drop=True)
st.write(f"Dataset size after removing outliers: **{df.shape}**")

# ── Step 4: Missing Values ────────────────────────────────────────────────────
st.header("Step 4: Handling Missing Values")
st.write("For a lot of columns, NaN doesn't mean the data is unknown — it means the house **doesn't have that feature**.")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    st.write("No missing values found.")
else:
    fig, ax = plt.subplots(figsize=(12, 5))
    missing.head(20).plot(kind='bar', color='#F8B195', ax=ax)
    ax.set_title('Columns with Missing Values (Top 20)')
    ax.set_ylabel('Number of Missing')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.write(missing.head(20))

no_feature_str = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
                  'GarageType','GarageFinish','GarageQual','GarageCond',
                  'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']
for col in no_feature_str:
    df[col] = df[col].fillna('None')

no_feature_num = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
                  'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
for col in no_feature_num:
    df[col] = df[col].fillna(0)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

st.write(f"Missing values remaining: **{df.isnull().sum().sum()}**")

# ── Step 5: Feature Engineering ──────────────────────────────────────────────
st.header("Step 5: Feature Engineering")

st.subheader("Feature 1: TotalSF")
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['TotalSF'], df['SalePrice'], alpha=0.4, s=10, color='#ADD0B3')
ax.set_xlabel('Total Square Footage')
ax.set_ylabel('SalePrice')
ax.set_title(f'TotalSF vs SalePrice (r = {df["TotalSF"].corr(df["SalePrice"]):.2f})')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Feature 2: TotalBathrooms")
df['TotalBathrooms'] = (df['FullBath'] + df['BsmtFullBath'] + 0.5*df['HalfBath'] + 0.5*df['BsmtHalfBath'])
st.write(df['TotalBathrooms'].value_counts().sort_index())

st.subheader("Feature 3: HouseAge")
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['HouseAge'], df['SalePrice'], alpha=0.4, s=10, color='#ADD0B3')
ax.set_xlabel('House Age (years)')
ax.set_ylabel('SalePrice')
ax.set_title(f'HouseAge vs SalePrice (r = {df["HouseAge"].corr(df["SalePrice"]):.2f})')
plt.tight_layout()
st.pyplot(fig)

st.subheader("Feature 4: OverallScore")
df['OverallScore'] = df['OverallQual'] * df['OverallCond']
st.write(f"OverallScore range: {df['OverallScore'].min()} - {df['OverallScore'].max()}")

st.subheader("Feature 5: HasGarage, HasPool, Has2ndFloor")
df['HasGarage']   = (df['GarageArea'] > 0).astype(int)
df['HasPool']     = (df['PoolArea'] > 0).astype(int)
df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
st.write(f"Homes with garage: {df['HasGarage'].sum()} | pool: {df['HasPool'].sum()} | 2nd floor: {df['Has2ndFloor'].sum()}")

# ── Step 6: Feature Selection ─────────────────────────────────────────────────
st.header("Step 6: Feature Selection")

df_model = df.drop(['Id', 'SalePrice'], axis=1).copy()
y = np.log1p(df['SalePrice'])
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

st.write(f"Total features: {df_model.shape[1]}")

with st.spinner("Running Random Forest for feature importance..."):
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_temp.fit(df_model, y)

importances = pd.Series(rf_temp.feature_importances_, index=df_model.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
importances.head(20).plot(kind='barh', color='#6C5B7B', ax=ax)
ax.invert_yaxis()
ax.set_title('Top 20 Most Important Features')
ax.set_xlabel('Importance')
plt.tight_layout()
st.pyplot(fig)

top10 = importances.head(10).index.tolist()
st.write("**Top 10 features selected:**")
for i, feat in enumerate(top10, 1):
    st.write(f"{i}. {feat}")

# ── Step 7: Train/Val Split ───────────────────────────────────────────────────
st.header("Step 7: Train/Validation Split")

X = df_model[top10]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training samples: **{len(X_train)}** | Validation samples: **{len(X_val)}**")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

# ── Step 8: Models ────────────────────────────────────────────────────────────
st.header("Step 8: Build and Evaluate Models")

# Linear Regression
st.subheader("Model 1: Linear Regression")
st.write("The simplest model. Finds the best straight line through the data. No hyperparameters to tune.")

lr_model = LinearRegression()
scores = cross_val_score(lr_model, X_train_sc, y_train, scoring='neg_mean_squared_error', cv=5)
st.write(f"5-Fold CV RMSE: **{np.sqrt(-scores.mean()):.4f}**")

lr_model.fit(X_train_sc, y_train)
y_pred_lr = lr_model.predict(X_val_sc)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr   = r2_score(y_val, y_pred_lr)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (log)", f"{rmse_lr:.4f}")
col2.metric("R²", f"{r2_lr:.4f}")
col3.metric("RMSE ($)", f"${np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred_lr))):,.0f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(np.expm1(y_val)/1000, np.expm1(y_pred_lr)/1000, color='pink', alpha=0.5, s=15)
ax.plot([0, 800], [0, 800], '--', color='#B43757')
ax.set_xlabel('Actual Price ($k)')
ax.set_ylabel('Predicted Price ($k)')
ax.set_title('Linear Regression - Actual vs Predicted')
st.pyplot(fig)

st.info("Linear Regression has no hyperparameters to tune — the algorithm finds the best fit mathematically. To improve it we give it better features and scale the data.")

# KNN
st.subheader("Model 2: K-Nearest Neighbors (KNN)")
st.write("Predicts price by averaging the K most similar houses. We use cross-validation to find the best K.")

k_values = list(range(2, 26))
cv_scores_knn = []
with st.spinner("Finding best K..."):
    for k in k_values:
        scores = cross_val_score(KNeighborsRegressor(n_neighbors=k), X_train_sc, y_train,
                                 scoring='neg_mean_squared_error', cv=5)
        cv_scores_knn.append(np.sqrt(-scores.mean()))

best_k = k_values[np.argmin(cv_scores_knn)]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(k_values, cv_scores_knn, marker='o', color='teal', markersize=5)
ax.axvline(best_k, color='#B43757', linestyle='--', label=f'Best K = {best_k}')
ax.set_xlabel('K')
ax.set_ylabel('CV RMSE')
ax.set_title('KNN - Finding Best K')
ax.legend()
st.pyplot(fig)
st.write(f"Best K: **{best_k}**")

knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_sc, y_train)
y_pred_knn = knn.predict(X_val_sc)
rmse_knn = np.sqrt(mean_squared_error(y_val, y_pred_knn))
r2_knn   = r2_score(y_val, y_pred_knn)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (log)", f"{rmse_knn:.4f}")
col2.metric("R²", f"{r2_knn:.4f}")
col3.metric("RMSE ($)", f"${np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred_knn))):,.0f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(np.expm1(y_val)/1000, np.expm1(y_pred_knn)/1000, alpha=0.5, s=15, color='teal')
ax.plot([0, 800], [0, 800], 'r--')
ax.set_xlabel('Actual Price ($k)')
ax.set_ylabel('Predicted Price ($k)')
ax.set_title('KNN - Actual vs Predicted')
st.pyplot(fig)

# Random Forest
st.subheader("Model 3: Random Forest")
st.write("Builds many decision trees and averages their predictions. We tune the number of trees and max depth.")

grid_results = []
with st.spinner("Tuning Random Forest (this takes a moment)..."):
    best_rf_rmse = 999
    best_rf_params = {}
    for n_trees in [50, 100, 200]:
        for max_d in [None, 10, 20]:
            scores = cross_val_score(
                RandomForestRegressor(n_estimators=n_trees, max_depth=max_d, random_state=42),
                X_train, y_train, scoring='neg_mean_squared_error', cv=5
            )
            rmse = np.sqrt(-scores.mean())
            grid_results.append({'n_trees': n_trees, 'max_depth': str(max_d), 'CV RMSE': round(rmse, 5)})
            if rmse < best_rf_rmse:
                best_rf_rmse = rmse
                best_rf_params = {'n_estimators': n_trees, 'max_depth': max_d}

st.dataframe(pd.DataFrame(grid_results))
st.write(f"Best params: **{best_rf_params}**")

rf = RandomForestRegressor(**best_rf_params, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf   = r2_score(y_val, y_pred_rf)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (log)", f"{rmse_rf:.4f}")
col2.metric("R²", f"{r2_rf:.4f}")
col3.metric("RMSE ($)", f"${np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_pred_rf))):,.0f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(np.expm1(y_val)/1000, np.expm1(y_pred_rf)/1000, alpha=0.5, s=15, color='#6C5B7B')
ax.plot([0, 800], [0, 800], 'r--')
ax.set_xlabel('Actual Price ($k)')
ax.set_ylabel('Predicted Price ($k)')
ax.set_title('Random Forest - Actual vs Predicted')
st.pyplot(fig)

# ── Step 9: Compare ───────────────────────────────────────────────────────────
st.header("Step 9: Compare All Models")

results = pd.DataFrame({
    'Model':    ['Linear Regression', 'KNN', 'Random Forest'],
    'RMSE_log': [rmse_lr, rmse_knn, rmse_rf],
    'R2':       [r2_lr,   r2_knn,   r2_rf]
})
st.dataframe(results.round(4))

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(results['Model'], results['RMSE_log'], color=['pink', 'teal', '#6C5B7B'])
    ax.set_title('RMSE (lower is better)')
    ax.set_ylabel('RMSE (log scale)')
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(results['Model'], results['R2'], color=['pink', 'teal', '#6C5B7B'])
    ax.set_title('R2 Score (higher is better)')
    ax.set_ylabel('R2')
    ax.set_ylim(0, 1)
    st.pyplot(fig)

st.subheader("Predicted vs Actual Prices")
results_df = pd.DataFrame({
    'Actual Price':    np.expm1(y_val.values),
    'Linear Reg Pred': np.expm1(y_pred_lr),
    'KNN Pred':        np.expm1(y_pred_knn),
    'RF Pred':         np.expm1(y_pred_rf)
})
st.dataframe(results_df.head(10).round(0))

output_csv = pd.DataFrame({'Id': range(1, len(y_val)+1), 'SalePrice': np.expm1(y_pred_rf)})
st.download_button("⬇️ Download RF Predictions CSV",
                   output_csv.to_csv(index=False).encode('utf-8'),
                   "rf_predictions.csv", "text/csv")

# ── Step 10: Conclusion ───────────────────────────────────────────────────────
st.header("Step 10: Conclusion")
st.markdown(f"""
**Results:**
- Linear Regression: RMSE ≈ {rmse_lr:.2f}, R² ≈ {r2_lr:.2f}
- KNN: RMSE ≈ {rmse_knn:.2f}, R² ≈ {r2_knn:.2f}
- Random Forest: RMSE ≈ {rmse_rf:.2f}, R² ≈ {r2_rf:.2f}

In this project we:
1. Loaded and explored the House Prices dataset
2. Handled missing values using domain knowledge (NaN = feature doesn't exist)
3. Removed outliers that would mess up the models
4. Created 5 new features: TotalSF, TotalBathrooms, HouseAge, OverallScore, and binary flags
5. Used Random Forest feature importance to pick the top 10 most useful features
6. Trained 3 models: Linear Regression, KNN, and Random Forest
7. Evaluated each model on data it had never seen before using an 80/20 split

**Best model:** Random Forest — it handles complex patterns that Linear Regression can't because it's not limited to straight-line relationships.

**What I learned:**
- Cleaning the data and creating good features matters a lot, maybe even more than which model you pick
- Log-transforming a skewed target like SalePrice helps all models perform better
- Cross-validation helps us check the model without looking at test data
- Linear Regression is a great starting point because it's simple and easy to understand
- Visualizing data is extremely helpful to make informed decisions
""")