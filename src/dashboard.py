import streamlit as st
from data_processing import load_data, transform_file, category_totals, period_totals, cash_flow

# =========================================================
# DATA LOADING
# =========================================================

# Load data using the function from data_processing module
df = load_data()
df_processed = transform_file()

# =========================================================
# DASHBOARD
# =========================================================

# App title
st.title("Banking Dashboard")

# =========================================================
# RAW DATA TABLE
# =========================================================

# Processed data table
st.header("Transaction Data (Processed)")
st.dataframe(df_processed.head(20))


# =========================================================
# CHARTS
# =========================================================

# Placeholder for Cash Flow chart
st.header("Net Income/Expense per Month (CF)")
if not df.empty:
    monthly_cf = cash_flow(df_processed, period="M")
    st.line_chart(monthly_cf.set_index("month"))
else:
    st.write("No data to display.")


# =========================================================
# TABLES
# =========================================================

# Placeholder for Category totals
st.header("Sum per Category (Total)")
if not df_processed.empty:
    category_sum = category_totals(df_processed)
    st.dataframe(category_sum)
else:
    st.write("No processed data to display.")


# Placeholder for Period totals
st.header("Sum per Category (Monthly)")
if not df_processed.empty:
    period_sum = period_totals(df_processed, period="M")
    st.dataframe(period_sum)
else:
    st.write("No processed data to display.")
