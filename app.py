# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="🛒 E-commerce Recommender",
    page_icon="🧩",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>🛒 E-commerce Recommendation System</h1>
    <h4 style='text-align: center;'>Collaborative Filtering | Content-Based Filtering | Clustering Insights | Evaluation</h4>
""", unsafe_allow_html=True)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def preprocess_data(df):
    """Standardize column names and rename common ones."""
    df.columns = [col.strip().lower() for col in df.columns]
    rename_map = {}
    if "userid" in df.columns:
        rename_map["userid"] = "user_id"
    if "productid" in df.columns:
        rename_map["productid"] = "item_id"
    if "rating" in df.columns:
        rename_map["rating"] = "rating"
    df.rename(columns=rename_map, inplace=True)

    # Ensure only needed columns
    required = ["user_id", "item_id", "rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None
    return df

@st.cache_data
def build_user_item_matrix(df, user_col="user_id", item_col="item_id", rating_col="rating", sample_size=5000):
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    pivot = df.pivot_table(index=user_col, columns=item_col, values=rating_col, fill_value=0)
    return pivot

@st.cache_data
def compute_item_features(df, item_col="item_id", rating_col="rating"):
    df = df.dropna(subset=[item_col, rating_col]).copy()
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
    item_agg = df.groupby(item_col)[rating_col].agg(['mean', 'count', 'std']).rename(
        columns={'mean': 'avg_rating', 'count': 'rating_count', 'std': 'rating_std'}
    )
    item_agg['rating_std'] = item_agg['rating_std'].fillna(0)
    item_agg['high_ratings'] = df[df[rating_col] >= df[rating_col].quantile(0.8)].groupby(item_col)[rating_col].count()
    item_agg['low_ratings'] = df[df[rating_col] <= df[rating_col].quantile(0.2)].groupby(item_col)[rating_col].count()
    item_agg[['high_ratings', 'low_ratings']] = item_agg[['high_ratings', 'low_ratings']].fillna(0)
    scaler = MinMaxScaler()
    item_agg['popularity'] = scaler.fit_transform(item_agg[['rating_count']])
    return item_agg.fillna(0)

def item_content_similarity_matrix(item_features_df):
    feat_matrix = item_features_df.values
    sim = cosine_similarity(feat_matrix)
    sim_df = pd.DataFrame(sim, index=item_features_df.index, columns=item_features_df.index)
    return sim_df

def get_similar_items_by_content(sim_df, chosen_item, top_n=10):
    if chosen_item not in sim_df.index:
        return pd.Series(dtype=float)
    return sim_df[chosen_item].sort_values(ascending=False).drop(chosen_item).head(top_n)

def get_top_n_recommendations_user_based(sim_matrix, pivot, user_index, n=5):
    user_ratings = pivot.iloc[user_index]
    sim_scores = sim_matrix[user_index]
    weighted = sim_scores.dot(pivot.fillna(0))
    denom = sim_scores.sum()
    scores = weighted / denom if denom != 0 else weighted
    unrated = user_ratings[user_ratings == 0].index
    recs = pd.Series(scores, index=pivot.columns)[unrated].sort_values(ascending=False).head(n)
    return recs

@st.cache_data
def perform_clustering(pivot, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(pivot)
    cluster_df = pd.DataFrame({"user_id": pivot.index, "cluster": labels})
    return cluster_df

# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to", [
        "📂 Upload Data",
        "📊 Data Overview",
        "🤖 Collaborative Filtering",
        "🧠 Content-Based Filtering",
        "🧩 Clustering Insights",
        "📈 Model Evaluation"
    ])

    # ----------------------------
    # PAGE 1: Upload Data
    # ----------------------------
    if page == "📂 Upload Data":
        st.header("📂 Upload Your Dataset (CSV)")
        uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = preprocess_data(df)
            if df is not None:
                st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                st.session_state["data"] = df
                st.write("Sample Data:")
                st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("Please upload a CSV file with columns: `user_id`, `item_id`, and `rating`.")

    # Check data availability for other pages
    if "data" not in st.session_state:
        st.warning("⚠️ Please upload data first using the **'📂 Upload Data'** tab.")
        return

    df = st.session_state["data"]
    user_col, item_col, rating_col = "user_id", "item_id", "rating"

    # ----------------------------
    # PAGE 2: Data Overview
    # ----------------------------
    if page == "📊 Data Overview":
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ratings", len(df))
        col2.metric("Unique Users", df[user_col].nunique())
        col3.metric("Unique Items", df[item_col].nunique())

        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Rating Distribution")
        fig = px.histogram(df, x=rating_col, nbins=10, title="Rating Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # PAGE 3: Collaborative Filtering
    # ----------------------------
    elif page == "🤖 Collaborative Filtering":
        st.header("🤖 Collaborative Filtering (User-based)")
        sample_size = st.slider("Sample size", 1000, min(20000, len(df)), 5000, 500)
        pivot = build_user_item_matrix(df, user_col, item_col, rating_col, sample_size=sample_size)
        st.write(f"Matrix shape: {pivot.shape}")

        sim_matrix = cosine_similarity(pivot)
        st.success("✅ User similarity computed")

        user_index = st.slider("Select user index", 0, len(pivot) - 1, 0)
        n_recs = st.slider("Number of recommendations", 3, 20, 5)
        if st.button("Get Recommendations"):
            recs = get_top_n_recommendations_user_based(sim_matrix, pivot, user_index, n=n_recs)
            st.write("Top recommendations:")
            st.table(recs)

    # ----------------------------
    # PAGE 4: Content-Based Filtering
    # ----------------------------
    elif page == "🧠 Content-Based Filtering":
        st.header("🧠 Content-Based Filtering")
        item_feat = compute_item_features(df, item_col=item_col, rating_col=rating_col)
        st.subheader("Item Features")
        st.dataframe(item_feat.head(10))

        sim_df = item_content_similarity_matrix(item_feat)
        st.success("✅ Item similarity matrix ready")

        chosen_item = st.selectbox("Choose an item", options=list(sim_df.index))
        top_k = st.slider("Top-K similar items", 3, 20, 10)

        if st.button("Get Similar Items"):
            similar_items = get_similar_items_by_content(sim_df, chosen_item, top_k)
            st.table(similar_items)

            fig = px.bar(similar_items.reset_index(), x="index", y=chosen_item,
                         title=f"Top {top_k} similar items to {chosen_item}")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # PAGE 5: Clustering Insights
    # ----------------------------
    elif page == "🧩 Clustering Insights":
        st.header("🧩 Clustering (KMeans) on Users")
        pivot = build_user_item_matrix(df, user_col, item_col, rating_col, sample_size=3000)
        n_clusters = st.slider("Number of clusters", 2, 12, 5)
        if st.button("Run Clustering"):
            cluster_df = perform_clustering(pivot, n_clusters)
            st.dataframe(cluster_df.head(10))
            cluster_counts = cluster_df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig = px.bar(cluster_counts, x="Cluster", y="Count", title="Cluster Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # PAGE 6: Model Evaluation
    # ----------------------------
    elif page == "📈 Model Evaluation":
        st.header("📈 Model Evaluation (Example)")
        results = pd.DataFrame({
            "Model": ["User-CF", "Item-CF", "Content-Based"],
            "Precision@10": [0.64, 0.68, 0.71],
            "Recall@10": [0.56, 0.61, 0.67],
            "F1@10": [0.59, 0.64, 0.69]
        })
        st.dataframe(results, use_container_width=True)
        fig = px.bar(results.melt(id_vars="Model"), x="Model", y="value", color="variable", barmode="group",
                     title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)


# Run app
if __name__ == "__main__":
    main()
