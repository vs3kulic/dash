import streamlit as st
from data_processing import load_data

# Load data using the function from data_processing module
df = load_data()

# App title
st.title("Banking Dashboard")

# Display raw data table
st.header("Transaction Data (Raw)")
st.dataframe(df.head(20))

# Placeholder for a simple chart (e.g., amount over time)
st.header("Net Income/Expense Over Time (CF)")
if not df.empty:
    # Group by month and sum amounts
    chart_data = df.groupby(df['booking_date'].dt.to_period('M'))['amount'].sum().reset_index()
    chart_data['booking_date'] = chart_data['booking_date'].dt.to_timestamp()
    st.line_chart(chart_data.set_index('booking_date'))
else:
    st.write("No data to display.")
