import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Refund Analyzer", layout="wide")

st.title("💸 Refund Analyzer Dashboard")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload Refund Data (Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Refund value calculation
    df["refund_value"] = df.apply(
        lambda row: row["sale_value"] if pd.notna(row["refund_comment"]) else 0,
        axis=1,
    )

    # ---------------------------
    # Summary Metrics
    # ---------------------------
    total_orders = len(df)
    total_refund_value = df["refund_value"].sum()
    refund_orders = df[df["refund_value"] > 0].shape[0]
    refund_percent = round((refund_orders / total_orders) * 100, 2) if total_orders > 0 else 0
    total_refund_days = df[df["refund_value"] > 0]["order_date"].nunique()
    top_refund_sku = df.groupby("skuid")["refund_value"].sum().sort_values(ascending=False).head(1)
    high_risk_customers = df.groupby("customer_id")["refund_value"].sum()
    high_risk_customers = high_risk_customers[high_risk_customers > high_risk_customers.mean()].count()

    # Display Summary Cards
    st.subheader("📊 Summary Stats")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("💰 Total Refund Value", f"₹{total_refund_value:,.0f}")
    col2.metric("📉 Refund % of Orders", f"{refund_percent}%")
    col3.metric("📅 Refund Days", f"{total_refund_days}")
    col4.metric("🥛 Top Refund SKU", f"{top_refund_sku.index[0]} (₹{top_refund_sku.values[0]:,.0f})" if not top_refund_sku.empty else "N/A")
    col5.metric("👤 High-Risk Customers", f"{high_risk_customers}")

    # ---------------------------
    # Multiple Chart Views
    # ---------------------------
    st.subheader("📈 Refund Analysis by Dimension")

    chart_option = st.selectbox("Select Dimension:", ["Hub", "dark_store", "category", "society_id"])

    fig, ax = plt.subplots()
    df.groupby(chart_option)["refund_value"].sum().sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_ylabel("Refund Value (₹)")
    ax.set_title(f"Refunds by {chart_option}")
    st.pyplot(fig)

    # Fraud Customer Chart
    st.subheader("🚨 Potential Fraud Customers")
    fraud_customers = df.groupby("customer_id")["refund_value"].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    fraud_customers.plot(kind="bar", ax=ax2, color="red")
    ax2.set_ylabel("Refund Value (₹)")
    ax2.set_title("Top 10 High Refund Customers")
    st.pyplot(fig2)

    # ---------------------------
    # AI Insights Section
    # ---------------------------
    st.subheader("🤖 AI Insights")
    insights = f"""
    - Refunds are concentrated in **{chart_option}** with highest losses.  
    - Refund percentage of orders is **{refund_percent}%**, which is {"high 🚨" if refund_percent > 5 else "under control ✅"}.  
    - SKU `{top_refund_sku.index[0] if not top_refund_sku.empty else "N/A"}` is driving majority of refund cost.  
    - Identified **{high_risk_customers} high-risk customers** who may need blocking or closer monitoring.  
    """
    st.text_area("AI Analysis", insights, height=150)

    # ---------------------------
    # Recommendations Panel
    # ---------------------------
    st.subheader("🛠 Recommendations Panel")
    recommendations = [
        "📌 Block customers crossing refund % threshold (e.g., > 5% of their orders).",
        "🔍 Investigate Top Refund SKU root causes (quality, packaging, delivery delays).",
        "🚦 Apply Refund Blocking Criteria: Red = >₹250 & >3 Refund Days, Yellow = Moderate, Green = Safe.",
        "🏷 Track refunds by Hub & Dark Store to isolate operational issues.",
        "🤝 Educate customers on proper storage/handling for Milk & FMCG to reduce claims."
    ]
    for rec in recommendations:
        st.write(f"- {rec}")
