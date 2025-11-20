import streamlit as st
from data_processing import load_data, transform_file

# Load data using the function from data_processing module
df = load_data()
df_processed = transform_file()

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

# Placeholder for a more complex chart (e.g., amount over time)
st.header("Expense per Category")
if not df_processed.empty:
    # Exclude internal_transfer category
    df_temp = df_processed[df_processed['category'] != 'internal_transfer'].copy()
    # Exclude positive amounts (income)
    df_temp = df_temp[df_temp['amount'] < 0]
    # Group by month and category
    df_temp['month'] = df_temp['booking_date'].dt.to_period('M')
    chart_data_cat = df_temp.groupby(['month', 'category'])['amount'].sum().reset_index()
    chart_data_cat['month'] = chart_data_cat['month'].dt.to_timestamp()
    chart_data_cat = chart_data_cat.pivot(index='month', columns='category', values='amount').fillna(0)
    st.bar_chart(chart_data_cat)
else:
    st.write("No data to display.")