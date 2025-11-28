import streamlit as st
from data_processing import load_data, transform_file, category_totals, period_totals

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
# DATA TABLES
# =========================================================

# Processed data table
st.header("Transaction Data (Processed)")
st.dataframe(df_processed.head(20))


# =========================================================
# CHARTS
# =========================================================

# Placeholder for a simple chart (e.g., amount over time)
st.header("Net Income/Expense per Month (CF)")
if not df.empty:
    # Group by month and sum amounts
    chart_data = df.groupby(df['booking_date'].dt.to_period('M'))['amount'].sum().reset_index()
    chart_data['booking_date'] = chart_data['booking_date'].dt.to_timestamp()
    st.line_chart(chart_data.set_index('booking_date'))
else:
    st.write("No data to display.")


# Placeholder for processed data table
st.header("Sum per Category (Total)")
if not df_processed.empty:
    category_sum = category_totals(df_processed)
    st.dataframe(category_sum)
else:
    st.write("No processed data to display.")


# Placeholder for period totals
st.header("Sum per Category (Monthly)")
if not df_processed.empty:
    period_sum = period_totals(df_processed, "M")
    st.dataframe(period_sum)
else:
    st.write("No processed data to display.")

