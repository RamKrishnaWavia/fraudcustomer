import io
import os
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.express as px

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# OpenAI (new SDK style)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # graceful fallback if package missing

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Refund Analyzer (AI)", layout="wide")
st.title("💸 Refund Analyzer Dashboard (AI)")

# ---------------------------
# Template (download)
# ---------------------------
TEMPLATE_COLUMNS = [
    "Report_date","customer_id","bb_id","order_id","TYPE","DC_name","fc_name","fc_id","dark_store",
    "category","Hub","society_name","sub_category","skuid","brand","product_name","pack_size",
    "hsn","GST_percent","Cess_percent","sales_without_tax","SGST_Value","CGST_Value","IGST_Value",
    "CESS_Value","order_date","quantity","mrp","cost_price","selling_price","sale_value",
    "slot_charges","is_indent","refund_comment","subscription_id","sales_without_delivery_charge",
    "discount_amount","is_free_coupon_product","user_source","delivery_status","society_id",
    "block_name","tag","order_ver","bb_order_id","fo_customer","cost_price_processed"
]

with st.sidebar:
    st.subheader("⬇️ Download Input Template")
    template_df = pd.DataFrame(columns=TEMPLATE_COLUMNS)
    bio = io.BytesIO()
    template_df.to_excel(bio, index=False, sheet_name="Template")
    bio.seek(0)
    st.download_button(
        "📥 Excel Template",
        data=bio,
        file_name="refund_analyzer_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Fill columns. A row is a refund if `refund_comment` is non-empty; refund value = `sale_value`.")

# ---------------------------
# File Upload
# ---------------------------
st.subheader("📤 Upload Data")
uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

def _safe_dt(series):
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def ai_write(insights_prompt:str) -> str:
    """Call OpenAI to write insights. Falls back to prompt echo if unavailable."""
    key = st.secrets.get("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return "⚠️ AI key not set. Add `OPENAI_API_KEY` in Streamlit Secrets to enable dynamic insights."
    try:
        client = OpenAI(api_key=key)
        # Use Chat Completions (official) or Responses; here we use chat for wide compatibility.
        # See OpenAI API reference for details.  # docs: platform.openai.com/docs
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a senior analytics consultant. Be concise, data-driven, and actionable."},
                {"role":"user","content":insights_prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI call failed: {e}"

def export_pdf(title:str, insights:str, recs:str) -> bytes:
    """Make a minimal PDF with insights + recommendations."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def draw_wrapped(text, x, y, max_width):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        lines = []
        for para in text.split("\n"):
            words = para.split(" ")
            line = ""
            for w in words:
                test = (line + " " + w).strip()
                if stringWidth(test, "Helvetica", 10) <= max_width:
                    line = test
                else:
                    lines.append(line)
                    line = w
            lines.append(line)
        for ln in lines:
            c.drawString(x, y, ln)
            y -= 12
            if y < 2*cm:
                c.showPage(); y = height - 2.5*cm
        return y

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2.5*cm, title)
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, height - 3.1*cm, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y = height - 4.2*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "AI Insights"); y -= 16
    c.setFont("Helvetica", 10); y = draw_wrapped(insights, 2*cm, y, width - 4*cm)

    y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Recommendations"); y -= 16
    c.setFont("Helvetica", 10); y = draw_wrapped(recs, 2*cm, y, width - 4*cm)

    c.showPage(); c.save()
    buf.seek(0)
    return buf.getvalue()

if not uploaded:
    st.info("Upload the Excel file to start.")
else:
    # ---------------------------
    # Load & Prepare
    # ---------------------------
    df = pd.read_excel(uploaded)
    # Normalize likely datetime columns
    for col in ["Report_date","order_date"]:
        if col in df.columns:
            df[col] = _safe_dt(df[col])

    # Refund flag & value
    df["is_refund"] = df["refund_comment"].astype(str).str.strip().ne("") & df["refund_comment"].notna()
    df["refund_value"] = df["sale_value"].where(df["is_refund"], 0)

    refund_df = df[df["is_refund"]].copy()

    # ---------------------------
    # Summary Cards
    # ---------------------------
    total_orders = len(df)
    total_refund_orders = len(refund_df)
    total_refund_value = float(refund_df["refund_value"].sum())
    refund_percent_of_orders = (total_refund_orders / total_orders * 100) if total_orders else 0.0
    refund_days = refund_df["Report_date"].dt.date.nunique() if "Report_date" in refund_df else 0
    top_sku_row = refund_df.groupby(["product_name","skuid"], dropna=False)["refund_value"].sum().sort_values(ascending=False).head(1)
    if len(top_sku_row) > 0:
        (top_prod, top_sku), top_val = top_sku_row.index[0], float(top_sku_row.iloc[0])
        top_sku_label = f"{top_prod} (SKU {top_sku}) – ₹{top_val:,.0f}"
    else:
        top_sku_label = "N/A"
    # High-risk customers: refund% > 20% and >= 3 refunds (tweakable)
    cust_agg = refund_df.groupby("customer_id").agg(
        refunds=("refund_value","sum"),
        refund_count=("refund_value","size"),
    ).merge(
        df.groupby("customer_id").agg(total_spend=("sale_value","sum")), left_index=True, right_index=True, how="left"
    )
    cust_agg["refund_pct_value"] = (cust_agg["refunds"] / cust_agg["total_spend"].replace(0, pd.NA))*100
    high_risk = cust_agg[(cust_agg["refund_count"]>=3) & (cust_agg["refund_pct_value"]>=20)].shape[0]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Refund Value", f"₹{total_refund_value:,.0f}")
    c2.metric("📊 Refund % of Orders", f"{refund_percent_of_orders:.2f}%")
    c3.metric("📅 Total Refund Days", f"{refund_days}")
    c4.metric("🥭 Top Refund SKU", top_sku_label)
    c5.metric("👤 High-Risk Customers", f"{high_risk}")

    st.divider()

    # ---------------------------
    # Charts (tabs)
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏘 Society", "🏢 Hub", "🏬 Dark Store", "🏷 Category", "🚨 Fraud Customers"]
    )

    def bar(df_, x, y, title):
        fig = px.bar(df_.sort_values(y, ascending=False), x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)

    with tab1:
        grp = refund_df.groupby("society_id", dropna=False)["refund_value"].sum().reset_index()
        bar(grp, "society_id", "refund_value", "Refund Value by Society")

    with tab2:
        grp = refund_df.groupby("Hub", dropna=False)["refund_value"].sum().reset_index()
        bar(grp, "Hub", "refund_value", "Refund Value by Hub")

    with tab3:
        grp = refund_df.groupby("dark_store", dropna=False)["refund_value"].sum().reset_index()
        bar(grp, "dark_store", "refund_value", "Refund Value by Dark Store")

    with tab4:
        grp = refund_df.groupby("category", dropna=False)["refund_value"].sum().reset_index()
        bar(grp, "category", "refund_value", "Refund Value by Category")

    with tab5:
        fraud_top = refund_df.groupby("customer_id")["refund_value"].sum().nlargest(10).reset_index()
        bar(fraud_top, "customer_id", "refund_value", "Top 10 High-Refund Customers")

    st.divider()

    # ---------------------------
    # AI Insights + Recommendations (dynamic)
    # ---------------------------
    st.subheader("🤖 AI Insights")
    # Build a compact summary for the model
    def df_head_to_text(df_, n=10):
        return df_.head(n).to_string(index=False)

    prompt = f"""
You are analyzing a grocery subscription/city logistics refund dataset.

Key metrics:
- Total orders: {total_orders}
- Refund orders: {total_refund_orders}
- Refund % of orders: {refund_percent_of_orders:.2f}%
- Total refund value (INR): {total_refund_value:,.0f}
- Refund days (unique Report_date): {refund_days}
- Top refund SKU: {top_sku_label}
- High-risk customers (>=3 refunds & >=20% refund value share): {high_risk}

Top aggregations (value in INR):
- By Category:
{refund_df.groupby('category', dropna=False)['refund_value'].sum().sort_values(ascending=False).head(10).to_string()}

- By Hub:
{refund_df.groupby('Hub', dropna=False)['refund_value'].sum().sort_values(ascending=False).head(10).to_string()}

- By Dark Store:
{refund_df.groupby('dark_store', dropna=False)['refund_value'].sum().sort_values(ascending=False).head(10).to_string()}

- By Society:
{refund_df.groupby('society_id', dropna=False)['refund_value'].sum().sort_values(ascending=False).head(10).to_string()}

Write:
1) 5 crisp insights (with % deltas or comparisons where useful).
2) 5 actionable recommendations (blocking criteria, SKU audits, ops fixes).
Keep it concise. Avoid boilerplate.
"""

    insights_md = ai_write(prompt)
    st.markdown(insights_md)

    st.subheader("✅ Recommendations Panel")
    recs_md = ai_write(
        "From the same data context, list 6 pragmatic actions ordered by ROI to reduce refunds next week. Be specific (thresholds, owners, timelines)."
    )
    st.markdown(recs_md)

    # ---------------------------
    # Exports: PDF + Excel
    # ---------------------------
    st.divider()
    st.subheader("⬇️ Download Reports")

    # Excel package of key tables
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as xl:
        df.to_excel(xl, index=False, sheet_name="All_Data")
        refund_df.to_excel(xl, index=False, sheet_name="Refund_Only")
        refund_df.groupby("category")["refund_value"].sum().reset_index().to_excel(xl, index=False, sheet_name="By_Category")
        refund_df.groupby("Hub")["refund_value"].sum().reset_index().to_excel(xl, index=False, sheet_name="By_Hub")
        refund_df.groupby("dark_store")["refund_value"].sum().reset_index().to_excel(xl, index=False, sheet_name="By_DarkStore")
        refund_df.groupby("society_id")["refund_value"].sum().reset_index().to_excel(xl, index=False, sheet_name="By_Society")
        refund_df.groupby("customer_id")["refund_value"].sum().reset_index().to_excel(xl, index=False, sheet_name="By_Customer")
    excel_buf.seek(0)
    st.download_button(
        "📊 Download Excel Pack",
        data=excel_buf,
        file_name=f"refund_report_{dt.date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # PDF (insights + recommendations)
    pdf_bytes = export_pdf(
        title="Refund Analyzer — AI Insights",
        insights=insights_md,
        recs=recs_md
    )
    st.download_button(
        "🧾 Download Insights PDF",
        data=pdf_bytes,
        file_name=f"refund_insights_{dt.date.today().isoformat()}.pdf",
        mime="application/pdf"
    )

