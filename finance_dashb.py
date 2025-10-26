import altair as alt
import numpy as np
import os
import io
import streamlit as st
from typing import Union


@st.cache_data
def load_data_from_csv(file_path_or_bytes: Union[str, bytes, None] = None):
    """Load and process CSV data. Accepts file path (str), raw bytes, or None (uses default file)."""
    # Resolve input to a pandas-readable object
    if file_path_or_bytes is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))       
        file_path_or_bytes = os.path.join(base_dir, 'financial_dataset.csv')

    if isinstance(file_path_or_bytes, (bytes, bytearray)):
        df = pd.read_csv(io.BytesIO(file_path_or_bytes))
    else:
        # assume string path or file-like; let pandas handle the error if wrong type
        df = pd.read_csv(file_path_or_bytes)

    # Ensure expected numeric columns exist (fill with 0 if missing)
    numeric_cols = [
        'Product Revenue', 'Service Revenue', 'Subscription Revenue', 'Other Revenue',
        'R&D', 'Sales & Marketing', 'General & Administrative', 'Operations', 'Cost of Goods Sold'
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    # Totals
    df['Total Revenue'] = (
        df['Product Revenue'] +
        df['Service Revenue'] +
        df['Subscription Revenue'] +
        df['Other Revenue']
    )
    df['Total Expenses'] = (
        df['R&D'] +
        df['Sales & Marketing'] +
        df['General & Administrative'] +
        df['Operations'] +
        df['Cost of Goods Sold']
    )

    # Monthly vs quarterly
    if 'Month' in df.columns:
        monthly_df = df[['Month', 'Total Revenue', 'Total Expenses']].copy()
        monthly_df.columns = ['Month', 'Revenue', 'Expenses']
        monthly_df['Profit'] = monthly_df['Revenue'] - monthly_df['Expenses']

        revenue_df = df.groupby('Quarter').agg({
            'Product Revenue': 'sum',
            'Service Revenue': 'sum',
            'Subscription Revenue': 'sum',
            'Other Revenue': 'sum'
        }).reset_index()
        revenue_df['Total Revenue'] = (
            revenue_df['Product Revenue'] +
            revenue_df['Service Revenue'] +
            revenue_df['Subscription Revenue'] +
            revenue_df['Other Revenue']
        )

        expense_df = df.groupby('Quarter').agg({
            'R&D': 'sum',
            'Sales & Marketing': 'sum',
            'General & Administrative': 'sum',
            'Operations': 'sum',
            'Cost of Goods Sold': 'sum'
        }).reset_index()
        expense_df['Total Expenses'] = (
            expense_df['R&D'] +
            expense_df['Sales & Marketing'] +
            expense_df['General & Administrative'] +
            expense_df['Operations'] +
            expense_df['Cost of Goods Sold']
        )
    else:
        # Quarterly-only: create monthly synthetic data deterministically
        revenue_df = df[['Quarter', 'Product Revenue', 'Service Revenue', 'Subscription Revenue', 'Other Revenue']].copy()
        revenue_df['Total Revenue'] = (
            revenue_df['Product Revenue'] +
            revenue_df['Service Revenue'] +
            revenue_df['Subscription Revenue'] +
            revenue_df['Other Revenue']
        )

        expense_df = df[['Quarter', 'R&D', 'Sales & Marketing', 'General & Administrative', 'Operations', 'Cost of Goods Sold']].copy()
        expense_df['Total Expenses'] = (
            expense_df['R&D'] +
            expense_df['Sales & Marketing'] +
            expense_df['General & Administrative'] +
            expense_df['Operations'] +
            expense_df['Cost of Goods Sold']
        )

        rng = np.random.default_rng(42)
        months = []
        monthly_revenue = []
        monthly_expenses = []
        for i in range(len(revenue_df)):
            base_revenue = revenue_df.loc[i, 'Total Revenue']
            base_expense = expense_df.loc[i, 'Total Expenses']
            for month in range(3):
                months.append(f"Month {i*3 + month + 1}")
                variation = rng.uniform(0.85, 1.15)
                monthly_revenue.append(base_revenue / 3 * variation)
                monthly_expenses.append(base_expense / 3 * variation)

        monthly_df = pd.DataFrame({
            'Month': months,
            'Revenue': monthly_revenue,
            'Expenses': monthly_expenses,
        })
        monthly_df['Profit'] = monthly_df['Revenue'] - monthly_df['Expenses']

    # Profit dataframe (quarterly)
    profit_df = pd.DataFrame({
        'Quarter': revenue_df['Quarter'],
        'Total Revenue': revenue_df['Total Revenue'],
        'Total Expenses': expense_df['Total Expenses'],
    })

    if 'Cost of Goods Sold' in expense_df.columns:
        profit_df['Gross Profit'] = profit_df['Total Revenue'] - expense_df['Cost of Goods Sold'].values
    else:
        profit_df['Gross Profit'] = profit_df['Total Revenue']

    profit_df['Operating Profit'] = profit_df['Total Revenue'] - profit_df['Total Expenses']
    profit_df['Profit Margin'] = profit_df.apply(
        lambda r: (r['Operating Profit'] / r['Total Revenue']) * 100 if r['Total Revenue'] else 0.0,
        axis=1,
    )

    return revenue_df, expense_df, profit_df, monthly_df


def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV (Streamlit UploadedFile) and return structured dataframes"""
    if uploaded_file is None:
        raise ValueError("No uploaded file provided")
    # Convert uploaded file to bytes so caching can hash it deterministically
    file_bytes = uploaded_file.getvalue()
    return load_data_from_csv(file_bytes)


def calculate_kpis(revenue_df: pd.DataFrame, profit_df: pd.DataFrame) -> dict:
    """Calculate key performance indicators with guards."""
    total_revenue = revenue_df['Total Revenue'].sum()
    total_profit = profit_df['Operating Profit'].sum()
    avg_profit_margin = profit_df['Profit Margin'].mean() if 'Profit Margin' in profit_df.columns else 0.0

    # growth rate (guard division by zero)
    if len(revenue_df) == 0:
        growth_rate = 0.0
    else:
        first_revenue = revenue_df.iloc[0]['Total Revenue']
        last_revenue = revenue_df.iloc[-1]['Total Revenue']
        growth_rate = ((last_revenue - first_revenue) / first_revenue) * 100 if first_revenue else 0.0

    return {
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'avg_profit_margin': avg_profit_margin,
        'growth_rate': growth_rate,
    }


def main():
    # Page config must be called before other Streamlit calls
    st.set_page_config(page_title="Tech Company Financial Dashboard 2025", page_icon="📊", layout="wide")
    st.title("📊 Tech Company Financial Dashboard 2025")

    # Sidebar - data source
    with st.sidebar:
        st.header("📂 Data Source")
        data_source = st.radio("Select Data Source", ["Use Default Dataset (3 years)", "Upload CSV File"])

        if data_source == "Upload CSV File":
            st.markdown("### Upload Your Financial Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    revenue_df, expense_df, profit_df, monthly_df = process_uploaded_csv(uploaded_file)
                    st.success("✅ File uploaded and processed")
                except Exception as e:
                    st.error(f"Error processing uploaded file: {e}")
                    try:
                        revenue_df, expense_df, profit_df, monthly_df = load_data_from_csv()
                        st.info("Loaded default dataset as fallback")
                    except Exception as e2:
                        st.error(f"Could not load default dataset: {e2}")
                        st.stop()
            else:
                st.info("No file uploaded - using default dataset")
                revenue_df, expense_df, profit_df, monthly_df = load_data_from_csv()
        else:
            revenue_df, expense_df, profit_df, monthly_df = load_data_from_csv()

    # Basic validation
    if len(revenue_df) == 0:
        st.error("No data available")
        st.stop()

    # KPIs
    kpis = calculate_kpis(revenue_df, profit_df)

    st.header("🎯 Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${kpis['total_revenue']:.1f}M", delta=f"{kpis['growth_rate']:.1f}%")
    c2.metric("Total Profit", f"${kpis['total_profit']:.1f}M", delta=f"{kpis['avg_profit_margin']:.1f}%")
    c3.metric("Avg Profit Margin", f"{kpis['avg_profit_margin']:.1f}%")
    c4.metric("Revenue Growth", f"{kpis['growth_rate']:.1f}%")

    st.markdown("---")

    # Revenue chart
    st.header("💰 Revenue Analysis")
    revenue_melt = revenue_df.melt(
        id_vars=['Quarter'],
        value_vars=['Product Revenue', 'Service Revenue', 'Subscription Revenue', 'Other Revenue'],
        var_name='Revenue Type',
        value_name='Revenue',
    )
    chart_revenue = alt.Chart(revenue_melt).mark_bar().encode(
        x=alt.X('Quarter:N', sort=list(revenue_df['Quarter'])),
        y=alt.Y('Revenue:Q', title='Revenue (Millions USD)'),
        color='Revenue Type:N',
    ).properties(height=400, title='Revenue Breakdown by Quarter')
    st.altair_chart(chart_revenue, use_container_width=True)

    # Expense chart
    st.header("💸 Expense Analysis")
    expense_melt = expense_df.melt(
        id_vars=['Quarter'],
        value_vars=['R&D', 'Sales & Marketing', 'General & Administrative', 'Operations', 'Cost of Goods Sold'],
        var_name='Expense Category',
        value_name='Expense',
    )
    chart_expense = alt.Chart(expense_melt).mark_bar().encode(
        x=alt.X('Quarter:N', sort=list(expense_df['Quarter'])),
        y=alt.Y('Expense:Q', title='Expenses (Millions USD)'),
        color='Expense Category:N',
    ).properties(height=400, title='Expense Breakdown by Quarter')
    st.altair_chart(chart_expense, use_container_width=True)

    st.markdown("---")

    # Data tables
    st.header("📋 Detailed Financial Data")
    tab1, tab2, tab3 = st.tabs(["Revenue Data", "Expense Data", "Profit & Loss"])

    with tab1:
        st.dataframe(revenue_df, use_container_width=True)

    with tab2:
        st.dataframe(expense_df, use_container_width=True)

    with tab3:
        st.dataframe(profit_df, use_container_width=True)


if __name__ == '__main__':
    main()
