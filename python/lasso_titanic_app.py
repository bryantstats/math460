# =============================================
# Interactive LASSO vs Logistic Regression App
# Titanic Dataset (Teaching Version)
# =============================================

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LassoCV, lasso_path, LogisticRegression
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="LASSO Teaching App", layout="wide")
st.title("🎯 LASSO Path & Coefficient Exploration – Titanic Dataset")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("titanic_train.csv")
    features = ['Pclass','Sex','Age','Fare','Embarked','SibSp','Parch']
    target = 'Survived'
    df = df[features + [target]]
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return df, features, target

df, features, target = load_data()

# --- TRAIN/TEST SPLIT ---
X = df[features]
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- PREPROCESSING ---
num_cols = ['Pclass','Age','Fare','SibSp','Parch']
cat_cols = ['Sex','Embarked']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

X_train_prepared = preprocessor.fit_transform(x_train)
X_test_prepared = preprocessor.transform(x_test)
num_names = num_cols
cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
feature_names = num_names + cat_names

# --- LASSO PATH ---
alphas, coefs, _ = lasso_path(X_train_prepared, y_train, alphas=np.logspace(-2, 1, 50))

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_alpha = st.sidebar.slider(
    "Choose Alpha (λ): Regularization Strength",
    float(alphas.min()), float(alphas.max()),
    float(np.median(alphas)),
    step=0.01
)
show_logistic = st.sidebar.checkbox("Show Logistic Regression Comparison", value=True)

# --- Fit models ---
lasso = Lasso(alpha=selected_alpha, max_iter=10000)
lasso.fit(X_train_prepared, y_train)
lasso_coefs = lasso.coef_

lasso_cv = LassoCV(alphas=np.logspace(-2, 1, 50), cv=5, random_state=42)
lasso_cv.fit(X_train_prepared, y_train)
best_alpha = lasso_cv.alpha_

# Logistic regression (for comparison)
logreg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
logreg.fit(X_train_prepared, y_train)
log_coefs = logreg.coef_.flatten()

# --- PLOT 1: LASSO PATH ---
fig_path = go.Figure()
for i, name in enumerate(feature_names):
    fig_path.add_trace(go.Scatter(
        x=alphas, y=coefs[i, :],
        mode='lines',
        name=name
    ))
fig_path.add_vline(x=selected_alpha, line_dash="dash", line_color="red",
                   annotation_text=f"α={selected_alpha:.3f}", annotation_position="top right")
fig_path.update_xaxes(type="log", title="Alpha (λ) – Regularization Strength (log scale)")
fig_path.update_yaxes(title="Coefficient Value")
fig_path.update_layout(
    title="📈 LASSO Path (Coefficient Shrinkage)",
    height=500,
    legend_title_text="Features"
)
st.plotly_chart(fig_path, use_container_width=True)

# --- PLOT 2: Coefficient Comparison ---
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "LASSO": lasso_coefs,
    "Logistic": log_coefs
})

if show_logistic:
    fig_bar = go.Figure(data=[
        go.Bar(name='LASSO', x=coef_df["Feature"], y=coef_df["LASSO"]),
        go.Bar(name='Logistic', x=coef_df["Feature"], y=coef_df["Logistic"])
    ])
else:
    fig_bar = px.bar(coef_df, x="Feature", y="LASSO", title="LASSO Coefficients Only")

fig_bar.update_layout(
    barmode='group',
    title="🔍 Coefficient Comparison (LASSO vs Logistic Regression)",
    xaxis_title="Feature",
    yaxis_title="Coefficient Value",
    height=500
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Summary Metrics ---
col1, col2 = st.columns(2)
col1.metric("Optimal α (from CV)", f"{best_alpha:.4f}")
col2.metric("Current α (slider)", f"{selected_alpha:.4f}")

st.markdown("---")
st.markdown("**Notes for Teaching:**")
st.markdown("""
- As α increases → more regularization → coefficients shrink toward 0.  
- Logistic regression (if shown) gives comparable directionality for classification.  
- The vertical red line marks the α value currently used in the model.  
- Use this slider to visually connect α strength with feature selection.
""")
