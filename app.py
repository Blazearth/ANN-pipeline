import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
import warnings
import os

warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pet Adoption Prediction Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #8b95a5;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0e6ed;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #c0c8d4;
    }

    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h2, h3 {
        color: #d0d8e4 !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea44, transparent);
        margin: 1.5rem 0;
    }

    .success-box {
        background: linear-gradient(135deg, #00b09b22, #96c93d22);
        border: 1px solid #00b09b55;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .danger-box {
        background: linear-gradient(135deg, #eb344422, #f8585822);
        border: 1px solid #eb344455;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Color Palette ───────────────────────────────────────────────────────────
COLORS = [
    "#667eea",
    "#764ba2",
    "#f093fb",
    "#4facfe",
    "#00f2fe",
    "#43e97b",
    "#fa709a",
    "#fee140",
    "#a18cd1",
    "#fbc2eb",
]
PLOTLY_TEMPLATE = "plotly_dark"


# ── Helper Functions ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "predict-pet-adoption-status-dataset",
        "pet_adoption_data.csv",
    )
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df):
    """Replicate the notebook's preprocessing pipeline."""
    df_processed = df.copy()

    # Encode categorical features
    le_pet = LabelEncoder()
    df_processed["PetType"] = le_pet.fit_transform(df_processed["PetType"])

    le_size = LabelEncoder()
    df_processed["Size"] = le_size.fit_transform(df_processed["Size"])

    # One-hot encode Breed and Color (drop_first to avoid multi-collinearity)
    df_processed = pd.get_dummies(df_processed, columns=["Breed", "Color"], drop_first=True)

    # Convert boolean columns to int
    bool_cols = df_processed.select_dtypes(include=["bool"]).columns.tolist()
    df_processed[bool_cols] = df_processed[bool_cols].astype(int)

    # Separate target
    target = df_processed["AdoptionLikelihood"]
    df_processed = df_processed.drop(columns=["AdoptionLikelihood"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_processed)

    return df_processed, X_scaled, target, scaler, le_pet, le_size


def build_and_train_model(X_train, y_train, epochs=50):
    """Build and train an ANN using TensorFlow Keras (matching the original notebook)."""
    import tensorflow as tf
    from tensorflow import keras

    n_features = X_train.shape[1]

    model = keras.Sequential([
        keras.layers.Dense(20, input_shape=(n_features,), activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    return model, history


# ── Load Data ───────────────────────────────────────────────────────────────
df_raw = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐾 Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Overview",
            "📊 Data Exploration",
            "📈 Visualizations",
            "🧠 Model Training",
            "🔮 Predict",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; opacity:0.6; font-size:0.75rem;">
        <p>Pet Adoption Prediction</p>
        <p>ANN & ML Project</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🐾 Pet Adoption Prediction Dashboard")
    st.markdown(
        "Predicting pet adoption likelihood using **Artificial Neural Networks** (ANN)"
    )
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pets", f"{len(df_raw):,}")
    col2.metric("Pet Types", df_raw["PetType"].nunique())
    col3.metric("Breeds", df_raw["Breed"].nunique())
    adopted = df_raw["AdoptionLikelihood"].sum()
    col4.metric("Adoption Rate", f"{adopted / len(df_raw) * 100:.1f}%")

    st.markdown("---")

    # Quick distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Pet Type Distribution")
        pet_counts = df_raw["PetType"].value_counts().reset_index()
        pet_counts.columns = ["PetType", "Count"]
        fig = px.pie(
            pet_counts,
            names="PetType",
            values="Count",
            color_discrete_sequence=COLORS,
            hole=0.45,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=350,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Adoption Likelihood")
        adopt_counts = (
            df_raw["AdoptionLikelihood"]
            .value_counts()
            .reset_index()
        )
        adopt_counts.columns = ["Likelihood", "Count"]
        adopt_counts["Label"] = adopt_counts["Likelihood"].map(
            {0: "Not Adopted", 1: "Adopted"}
        )
        fig = px.pie(
            adopt_counts,
            names="Label",
            values="Count",
            color_discrete_sequence=["#fa709a", "#43e97b"],
            hole=0.45,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=350,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Dataset preview
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True, height=390)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Exploration":
    st.markdown("# 📊 Data Exploration")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Info", "📊 Statistics", "🔍 Null Values", "📄 Full Data"]
    )

    with tab1:
        st.markdown("### Dataset Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df_raw.shape[0])
        col2.metric("Columns", df_raw.shape[1])
        col3.metric("Memory", f"{df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.markdown("#### Column Types")
        dtype_df = pd.DataFrame(
            {
                "Column": df_raw.columns,
                "Type": df_raw.dtypes.astype(str).values,
                "Non-Null": df_raw.notnull().sum().values,
                "Unique": [df_raw[c].nunique() for c in df_raw.columns],
            }
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Descriptive Statistics")
        st.dataframe(
            df_raw.describe().T.round(2),
            use_container_width=True,
        )

    with tab3:
        st.markdown("### Missing Values")
        null_df = pd.DataFrame(
            {
                "Column": df_raw.columns,
                "Null Count": df_raw.isnull().sum().values,
                "Null %": (df_raw.isnull().sum().values / len(df_raw) * 100).round(2),
            }
        )
        st.dataframe(null_df, use_container_width=True, hide_index=True)

        if df_raw.isnull().sum().sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning("⚠️ Missing values detected. Consider imputation strategies.")

    with tab4:
        st.markdown("### Complete Dataset")
        st.dataframe(df_raw, use_container_width=True, height=500)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown("# 📈 Visualizations")
    st.markdown("---")

    viz_tab = st.tabs(
        [
            "🥧 Distributions",
            "📊 Histograms",
            "🔥 Correlation",
            "📦 Box Plots",
        ]
    )

    # ── Distributions ────────────────────────────────────────────────────
    with viz_tab[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Breed Distribution")
            breed_counts = (
                df_raw.groupby("Breed").size().reset_index(name="Count")
            )
            fig = px.pie(
                breed_counts,
                names="Breed",
                values="Count",
                color_discrete_sequence=COLORS,
                hole=0.4,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Color Distribution")
            color_counts = (
                df_raw.groupby("Color").size().reset_index(name="Count")
            )
            fig = px.pie(
                color_counts,
                names="Color",
                values="Count",
                color_discrete_sequence=COLORS[3:],
                hole=0.4,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Size Distribution")
            size_counts = (
                df_raw.groupby("Size").size().reset_index(name="Count")
            )
            fig = px.bar(
                size_counts,
                x="Size",
                y="Count",
                color="Size",
                color_discrete_sequence=COLORS,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=400, margin=dict(t=30, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("### Vaccination Status")
            vac_counts = (
                df_raw["Vaccinated"]
                .value_counts()
                .reset_index()
            )
            vac_counts.columns = ["Vaccinated", "Count"]
            vac_counts["Label"] = vac_counts["Vaccinated"].map(
                {0: "Not Vaccinated", 1: "Vaccinated"}
            )
            fig = px.pie(
                vac_counts,
                names="Label",
                values="Count",
                color_discrete_sequence=["#fa709a", "#43e97b"],
                hole=0.4,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Histograms ───────────────────────────────────────────────────────
    with viz_tab[1]:
        num_cols = ["AgeMonths", "WeightKg", "TimeInShelterDays", "AdoptionFee"]
        cols = st.columns(2)
        for i, col_name in enumerate(num_cols):
            with cols[i % 2]:
                st.markdown(f"### {col_name}")
                fig = px.histogram(
                    df_raw,
                    x=col_name,
                    nbins=30,
                    color_discrete_sequence=[COLORS[i]],
                    template=PLOTLY_TEMPLATE,
                    marginal="box",
                )
                fig.update_layout(height=350, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

    # ── Correlation ──────────────────────────────────────────────────────
    with viz_tab[2]:
        st.markdown("### Feature Correlation Heatmap")
        numeric_df = df_raw.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE,
            aspect="auto",
        )
        fig.update_layout(height=600, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Box Plots ────────────────────────────────────────────────────────
    with viz_tab[3]:
        st.markdown("### Feature Distribution by Adoption Likelihood")
        feature = st.selectbox(
            "Select feature",
            ["AgeMonths", "WeightKg", "TimeInShelterDays", "AdoptionFee"],
        )
        plot_df = df_raw.copy()
        plot_df["Adopted"] = plot_df["AdoptionLikelihood"].map(
            {0: "Not Adopted", 1: "Adopted"}
        )
        fig = px.box(
            plot_df,
            x="Adopted",
            y=feature,
            color="Adopted",
            color_discrete_sequence=["#fa709a", "#43e97b"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(height=450, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Training":
    st.markdown("# 🧠 Model Training & Evaluation")
    st.markdown("---")

    # Settings
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    with col_s2:
        max_iter = st.slider("Max Iterations (epochs)", 50, 500, 200, step=50)

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Preprocessing data & training Keras ANN model…"):
            df_proc, X_scaled, y, scaler, le_pet, le_size = preprocess_data(df_raw)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )

            progress_bar = st.progress(0, text="Training ANN…")
            model, history = build_and_train_model(X_train, y_train, epochs=max_iter)
            progress_bar.progress(100, text="Training complete!")

            # Store in session
            st.session_state["model"] = model
            st.session_state["history"] = history
            st.session_state["scaler"] = scaler
            st.session_state["le_pet"] = le_pet
            st.session_state["le_size"] = le_size
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["feature_cols"] = df_proc.columns.tolist()

        st.success("✅ Keras ANN model trained successfully!")

    # Show results if model exists
    if "model" in st.session_state:
        model = st.session_state["model"]
        history = st.session_state["history"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        st.markdown("---")

        # ── Model Architecture ───────────────────────────────────────────
        st.markdown("### 🏗️ Model Architecture (Keras Sequential)")
        arch_cols = st.columns(4)
        arch_cols[0].markdown("**Layer 1**\n\nDense(20, relu)")
        arch_cols[1].markdown("**Layer 2**\n\nDense(50, relu)")
        arch_cols[2].markdown("**Layer 3**\n\nDense(20, relu)")
        arch_cols[3].markdown("**Layer 4**\n\nDense(1, sigmoid)")

        st.markdown("---")

        # ── Metrics ──────────────────────────────────────────────────────
        y_pred_raw = model.predict(X_test, verbose=0)
        y_pred = (y_pred_raw > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        st.markdown("### 📊 Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Precision", f"{prec:.2%}")
        m3.metric("Recall", f"{rec:.2%}")
        m4.metric("F1 Score", f"{f1:.2%}")

        st.markdown("---")

        col_cm, col_report = st.columns(2)

        with col_cm:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                labels=dict(x="Predicted", y="Actual"),
                x=["Not Adopted", "Adopted"],
                y=["Not Adopted", "Adopted"],
            )
            fig.update_layout(height=400, margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_report:
            st.markdown("### Classification Report")
            report = classification_report(
                y_test, y_pred, target_names=["Not Adopted", "Adopted"], output_dict=True
            )
            report_df = pd.DataFrame(report).T
            st.dataframe(
                report_df.round(2),
                use_container_width=True,
            )

        # ── ROC Curve ────────────────────────────────────────────────────
        st.markdown("### ROC Curve")
        y_proba = y_pred_raw.flatten()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC = {roc_auc:.3f}",
                line=dict(color="#667eea", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Training History (Loss + Accuracy) ───────────────────────────
        st.markdown("### Training History")
        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['loss'], mode='lines',
                name='Train Loss', line=dict(color='#fa709a', width=2),
            ))
            if 'val_loss' in history.history:
                fig.add_trace(go.Scatter(
                    y=history.history['val_loss'], mode='lines',
                    name='Val Loss', line=dict(color='#f093fb', width=2, dash='dash'),
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, title='Loss',
                xaxis_title='Epoch', yaxis_title='Loss',
                height=350, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with hist_col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['accuracy'], mode='lines',
                name='Train Accuracy', line=dict(color='#43e97b', width=2),
            ))
            if 'val_accuracy' in history.history:
                fig.add_trace(go.Scatter(
                    y=history.history['val_accuracy'], mode='lines',
                    name='Val Accuracy', line=dict(color='#4facfe', width=2, dash='dash'),
                ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, title='Accuracy',
                xaxis_title='Epoch', yaxis_title='Accuracy',
                height=350, margin=dict(t=40, b=10),
                yaxis_tickformat='.0%',
            )
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("# 🔮 Predict Adoption Likelihood")
    st.markdown("---")

    if "model" not in st.session_state:
        st.warning("⚠️ Please train the model first on the **🧠 Model Training** page.")
    else:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        le_pet = st.session_state["le_pet"]
        le_size = st.session_state["le_size"]
        feature_cols = st.session_state["feature_cols"]

        st.markdown("### Enter Pet Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            pet_type = st.selectbox("Pet Type", ["Bird", "Cat", "Dog", "Rabbit"])
            breed_options = {
                "Bird": ["Parakeet"],
                "Cat": ["Persian", "Siamese"],
                "Dog": ["Golden Retriever", "Labrador", "Poodle"],
                "Rabbit": ["Rabbit"],
            }
            breed = st.selectbox("Breed", breed_options.get(pet_type, []))
            color = st.selectbox("Color", ["Black", "Brown", "Gray", "Orange", "White"])

        with col2:
            age = st.number_input("Age (months)", 1, 200, 36)
            weight = st.number_input("Weight (kg)", 0.5, 50.0, 10.0, step=0.5)
            size = st.selectbox("Size", ["Small", "Medium", "Large"])

        with col3:
            vaccinated = st.selectbox("Vaccinated?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            health = st.selectbox("Health Condition", [0, 1], format_func=lambda x: "Healthy" if x == 0 else "Has Condition")
            shelter_days = st.number_input("Time in Shelter (days)", 0, 200, 30)
            adoption_fee = st.number_input("Adoption Fee ($)", 0, 600, 150)
            prev_owner = st.selectbox("Previous Owner?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown("---")

        if st.button("🔮 Predict Adoption", type="primary", use_container_width=True):
            # Build input row matching the trained feature columns
            pet_id = 9999  # placeholder
            pet_type_enc = le_pet.transform([pet_type])[0]
            size_enc = le_size.transform([size])[0]

            input_dict = {
                "PetID": pet_id,
                "PetType": pet_type_enc,
                "AgeMonths": age,
                "Size": size_enc,
                "WeightKg": weight,
                "Vaccinated": vaccinated,
                "HealthCondition": health,
                "TimeInShelterDays": shelter_days,
                "AdoptionFee": adoption_fee,
                "PreviousOwner": prev_owner,
            }

            # Create one-hot columns for breed and color
            all_breeds = ["Golden Retriever", "Labrador", "Parakeet", "Persian", "Poodle", "Rabbit", "Siamese"]
            all_colors = ["Black", "Brown", "Gray", "Orange", "White"]

            for b in all_breeds:
                col_name = f"Breed_{b}"
                if col_name in feature_cols:
                    input_dict[col_name] = 1 if breed == b else 0

            for c in all_colors:
                col_name = f"Color_{c}"
                if col_name in feature_cols:
                    input_dict[col_name] = 1 if color == c else 0

            # Build df with correct column order
            input_df = pd.DataFrame([input_dict])
            # Ensure all feature columns exist
            for fc in feature_cols:
                if fc not in input_df.columns:
                    input_df[fc] = 0
            input_df = input_df[feature_cols]

            # Scale
            input_scaled = scaler.transform(input_df)

            pred_raw = model.predict(input_scaled, verbose=0)[0][0]
            pred = 1 if pred_raw > 0.5 else 0
            proba = np.array([1 - pred_raw, pred_raw])

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            if pred == 1:
                st.markdown(
                    f"""
                <div class="success-box">
                    <h2 style="color:#43e97b; margin:0;">✅ Likely to be Adopted!</h2>
                    <p style="font-size:1.3rem; margin-top:8px; color: #d0d8e4;">
                        Confidence: <strong>{proba[1]:.1%}</strong>
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="danger-box">
                    <h2 style="color:#fa709a; margin:0;">❌ Unlikely to be Adopted</h2>
                    <p style="font-size:1.3rem; margin-top:8px; color: #d0d8e4;">
                        Confidence: <strong>{proba[0]:.1%}</strong>
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Probability bar
            st.markdown("### Probability Breakdown")
            prob_df = pd.DataFrame(
                {"Outcome": ["Not Adopted", "Adopted"], "Probability": proba}
            )
            fig = px.bar(
                prob_df,
                x="Outcome",
                y="Probability",
                color="Outcome",
                color_discrete_sequence=["#fa709a", "#43e97b"],
                template=PLOTLY_TEMPLATE,
                text=prob_df["Probability"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(
                height=300,
                margin=dict(t=30, b=10),
                showlegend=False,
                yaxis_tickformat=".0%",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
