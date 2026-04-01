import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Data generation (Bashandy simulation)
# ------------------------------------------------------------
def generate_bashandy_data():
    # Menu items (extracted from Menumisr)
    menu_data = [
        # Salads
        ("Pickles Box", 3.0, "Salads"), ("Pickled Eggplant & Pepper Box", 3.0, "Salads"),
        ("Fried Eggplant & Pepper Box", 3.0, "Salads"), ("Green Salad Box", 5.0, "Salads"),
        # Foul Sandwiches (abbreviated for brevity – include all 21 items from the earlier list)
        # We'll include a representative subset to keep the simulation manageable.
        # For full effect, you can copy the entire list from the previous conversation.
        # Here I'll include a few to illustrate; you can expand.
        ("Bashandi Double Foul", 4.0, "Foul Sandwiches"), ("Bashandi Foul", 3.5, "Foul Sandwiches"),
        ("Foul with Salsa", 4.0, "Foul Sandwiches"), ("Foul with Eggplant", 4.5, "Foul Sandwiches"),
        ("Foul with Boiled Eggs", 7.0, "Foul Sandwiches"), ("Foul with Butter", 4.5, "Foul Sandwiches"),
        ("Foul with Fries", 5.0, "Foul Sandwiches"), ("Foul with Falafel", 4.0, "Foul Sandwiches"),
        ("New Deluxe Foul", 5.5, "Foul Sandwiches"), ("Foul with Fried Eggs", 8.0, "Foul Sandwiches"),
        ("Alexandrian Foul", 4.0, "Foul Sandwiches"), ("Foul with Baba Ghannoug", 4.5, "Foul Sandwiches"),
        ("Foul with Pastrami", 8.0, "Foul Sandwiches"), ("Foul with Sausage", 8.0, "Foul Sandwiches"),
        ("Foul with Sojouk", 8.0, "Foul Sandwiches"),
        # Falafel Sandwiches
        ("Double Falafel", 4.0, "Falafel Sandwiches"), ("Falafel", 3.5, "Falafel Sandwiches"),
        ("Stuffed Falafel", 4.0, "Falafel Sandwiches"), ("Falafel with Eggs", 7.0, "Falafel Sandwiches"),
        ("Falafel with Fries", 5.0, "Falafel Sandwiches"), ("Falafel with Baba Ghannog", 4.5, "Falafel Sandwiches"),
        ("Falafel with Eggplant", 4.5, "Falafel Sandwiches"), ("Falafel with Cottage Cheese", 5.0, "Falafel Sandwiches"),
        # Potato Sandwiches
        ("Fries", 5.5, "Potato Sandwiches"), ("Fries with Mayonnaise", 6.0, "Potato Sandwiches"),
        ("Fries with Ketchup", 6.0, "Potato Sandwiches"), ("Fries with Baba Ghannoug", 7.0, "Potato Sandwiches"),
        ("Chips", 5.5, "Potato Sandwiches"), ("Fries with Eggplant", 6.0, "Potato Sandwiches"),
        ("Fries with Breadcrumbs", 6.0, "Potato Sandwiches"), ("Fries with Cottage Cheese", 6.0, "Potato Sandwiches"),
        ("Fries with Old Cheese", 6.0, "Potato Sandwiches"), ("Fries with Roomi Cheese", 7.0, "Potato Sandwiches"),
        ("Fries Panne", 5.0, "Potato Sandwiches"), ("Mashed Potatoes", 5.5, "Potato Sandwiches"),
        ("Fries with Mayonnaise and Ketchup", 6.5, "Potato Sandwiches"),
        # Varieties Sandwiches
        ("Omelette with Roomi", 7.0, "Varieties Sandwiches"), ("Omelette Pizza", 9.5, "Varieties Sandwiches"),
        ("Boiled Eggs", 5.5, "Varieties Sandwiches"), ("Plain Omelette", 6.0, "Varieties Sandwiches"),
        ("Eggs with Pastrami", 8.0, "Varieties Sandwiches"), ("Apache", 11.5, "Varieties Sandwiches"),
        ("Shakshouka", 5.0, "Varieties Sandwiches"), ("Dynamite", 7.0, "Varieties Sandwiches"),
        ("Eggplant", 4.5, "Varieties Sandwiches"), ("Fried Cheese", 7.0, "Varieties Sandwiches"),
        ("Cottage Cheese with Tomato", 5.5, "Varieties Sandwiches"), ("Moussaka'a", 4.5, "Varieties Sandwiches"),
        ("Baba Ghanoug", 4.5, "Varieties Sandwiches"), ("French Egga", 4.5, "Varieties Sandwiches"),
        # Boxes
        ("Plain Foul", 4.5, "Boxes"), ("Bashandi Mehweg Foul", 5.5, "Boxes"),
        ("Alexandrian Foul (Box)", 6.5, "Boxes"), ("Foul with Salsa (Box)", 6.5, "Boxes"),
        ("Bashandi Foul with Flaxseed Oil", 6.5, "Boxes"), ("Bashandi Foul with Olive Oil", 7.0, "Boxes"),
        ("Mashed Potato", 6.0, "Boxes"), ("Cottage Cheese", 6.0, "Boxes"),
        ("Baba Ghanoug (Box)", 7.0, "Boxes"), ("Shakshoka", 7.0, "Boxes"),
        ("Moussaka'a (Box)", 6.0, "Boxes"), ("Falafel Paste", 5.0, "Boxes"),
        # Varieties Orders
        ("Fries with Spices Packet", 5.5, "Varieties Orders"), ("Fries with Ketchup Packet", 6.5, "Varieties Orders"),
        ("Fries with Mayonnaise Packet", 6.5, "Varieties Orders"), ("Fries with Roomi Cheese Packet", 10.0, "Varieties Orders"),
        ("Plain Falafel (Piece)", 1.0, "Varieties Orders"), ("Stuffed Falafel (Piece)", 1.5, "Varieties Orders"),
        ("Amaty Falafel (Piece)", 0.5, "Varieties Orders")
    ]

    menu_items = pd.DataFrame(menu_data, columns=['item', 'base_price', 'category'])
    # Assign cost (40% of base price) and popularity (based on category)
    menu_items['cost_per_unit'] = menu_items['base_price'] * 0.4
    cat_pop = {'Salads':0.6, 'Foul Sandwiches':0.9, 'Falafel Sandwiches':0.85, 'Potato Sandwiches':0.7,
               'Varieties Sandwiches':0.6, 'Boxes':0.5, 'Varieties Orders':0.4}
    menu_items['popularity'] = menu_items['category'].map(cat_pop)
    min_price = menu_items['base_price'].min()
    max_price = menu_items['base_price'].max()
    menu_items['popularity'] = menu_items.apply(
        lambda row: row['popularity'] * (1 - (row['base_price'] - min_price) / max_price * 0.3), axis=1)
    menu_items['popularity'] = menu_items['popularity'].clip(0.2, 0.95)

    # Generate 90 days of data
    np.random.seed(42)
    days = 90
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    hours = range(8, 23)

    data = []
    for date in dates:
        is_ramadan = (date.month in [3, 4])
        for hour in hours:
            day_type = 'weekend' if date.dayofweek >= 5 else 'weekday'
            time_factor = 1.2 if hour in [12,13,14,19,20,21,22] else 1.0
            if is_ramadan:
                time_factor = 0.3 if hour < 18 else 1.5
            for _, row in menu_items.iterrows():
                item = row['item']
                base_price = row['base_price']
                cost = row['cost_per_unit']
                popularity = row['popularity']
                price = base_price + np.random.uniform(-2, 2)
                price = max(1, price)
                base_demand = popularity * 20 * time_factor
                base_demand *= 1.3 if day_type == 'weekend' else 0.8
                # Stronger elasticity: -1.5
                demand = base_demand - 1.5 * (price - base_price)
                demand = max(0, int(demand + np.random.normal(0, 3)))
                revenue = price * demand
                total_cost = cost * demand
                profit = revenue - total_cost
                data.append({
                    'date': date, 'hour': hour, 'day_type': day_type, 'item': item,
                    'price': price, 'demand': demand, 'revenue': revenue,
                    'cost': total_cost, 'profit': profit
                })
    df = pd.DataFrame(data)
    return df, menu_items

# ------------------------------------------------------------
# Analysis functions (same as before)
# ------------------------------------------------------------
def best_hours(df):
    return df.groupby('hour')['profit'].sum().sort_values(ascending=False).head(3)

def worst_menu_items(df, profit_share_threshold=0.02, profit_per_unit_threshold=3, demand_share_threshold=0.05):
    item_summary = df.groupby('item').agg({'demand': 'sum', 'profit': 'sum'}).reset_index()
    item_summary['profit_per_unit'] = item_summary['profit'] / item_summary['demand']
    total_profit = item_summary['profit'].sum()
    avg_demand = item_summary['demand'].mean()
    bad = item_summary[
        (item_summary['profit'] < profit_share_threshold * total_profit) |
        (item_summary['profit_per_unit'] < profit_per_unit_threshold) |
        (item_summary['demand'] < demand_share_threshold * avg_demand)
    ]
    return bad

def profit_leaks(df, margin_threshold=0.25, profit_per_unit_threshold=3):
    item_summary = df.groupby('item').agg({'revenue': 'sum', 'profit': 'sum', 'demand': 'sum'}).reset_index()
    item_summary['margin'] = item_summary['profit'] / item_summary['revenue']
    item_summary['profit_per_unit'] = item_summary['profit'] / item_summary['demand']
    leaks = item_summary[
        (item_summary['margin'] < margin_threshold) |
        (item_summary['profit_per_unit'] < profit_per_unit_threshold)
    ]
    return leaks[['item', 'margin', 'profit_per_unit', 'profit']]

def best_price_for_item_linear(df, item_name, menu_items):
    item_df = df[df['item'] == item_name].copy()
    item_df['hour_sin'] = np.sin(2 * np.pi * item_df['hour'] / 24)
    item_df['hour_cos'] = np.cos(2 * np.pi * item_df['hour'] / 24)
    item_df = pd.get_dummies(item_df, columns=['day_type'], drop_first=True)
    X = item_df[['price', 'hour_sin', 'hour_cos', 'day_type_weekend']]
    y = item_df['demand']
    model = LinearRegression()
    model.fit(X, y)
    cost = menu_items[menu_items['item'] == item_name]['cost_per_unit'].values[0]
    avg_hour_sin = item_df['hour_sin'].mean()
    avg_hour_cos = item_df['hour_cos'].mean()
    avg_weekend = item_df['day_type_weekend'].mean()
    intercept = model.intercept_
    coef_price = model.coef_[0]
    coef_hsin = model.coef_[1]
    coef_hcos = model.coef_[2]
    coef_weekend = model.coef_[3]
    other_effects = (coef_hsin * avg_hour_sin + coef_hcos * avg_hour_cos + coef_weekend * avg_weekend)
    if coef_price < 0:
        optimal_price = - (intercept + other_effects - cost * coef_price) / (2 * coef_price)
        base = menu_items[menu_items['item'] == item_name]['base_price'].values[0]
        min_price = max(base * 0.8, 1.0)
        max_price = min(base * 1.5, 25.0)
        optimal_price = np.clip(optimal_price, min_price, max_price)
    else:
        optimal_price = menu_items[menu_items['item'] == item_name]['base_price'].values[0]
    X_opt = X.copy()
    X_opt['price'] = optimal_price
    predicted_demand = model.predict(X_opt)
    predicted_profit = (optimal_price - cost) * predicted_demand
    total_predicted_profit = predicted_profit.sum()
    actual_total_profit = item_df['profit'].sum()
    profit_increase = total_predicted_profit - actual_total_profit
    return optimal_price, profit_increase

# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------
st.set_page_config(page_title="Restaurant Intelligence Dashboard", layout="wide")
st.title("🍽️ Restaurant Operational Intelligence System")
st.markdown("Upload your POS data to get pricing, menu, and time insights.")

# Sidebar for data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose data source", ["Use example (Bashandy)", "Upload CSV"])

df = None
menu_items = None

if data_source == "Use example (Bashandy)":
    with st.spinner("Generating simulated Bashandy data..."):
        df, menu_items = generate_bashandy_data()
    st.sidebar.success("Simulated data ready!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = ['date', 'hour', 'day_type', 'item', 'price', 'demand', 'cost', 'profit']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
            st.stop()
        # Build menu_items from the data
        menu_items = df.groupby('item').agg({'price': 'mean', 'cost': 'mean'}).reset_index()
        menu_items = menu_items.rename(columns={'price': 'base_price', 'cost': 'cost_per_unit'})
        st.sidebar.success("File loaded successfully!")

if df is not None and menu_items is not None:
    with st.expander("📊 Full Report", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 Best Hours")
            best = best_hours(df)
            st.dataframe(best.rename("Profit (EGP)"))
        with col2:
            st.subheader("⚠️ Worst Menu Items")
            worst = worst_menu_items(df)
            if worst.empty:
                st.write("No items underperform (within thresholds).")
            else:
                st.dataframe(worst[['item', 'profit', 'profit_per_unit']])

        st.subheader("💸 Profit Leaks")
        leaks = profit_leaks(df)
        if leaks.empty:
            st.write("No profit leaks detected.")
        else:
            st.dataframe(leaks)

        st.subheader("💰 Optimal Price Suggestions")
        items = df['item'].unique()
        price_data = []
        for item in items:
            best_price, increase = best_price_for_item_linear(df, item, menu_items)
            current_price = df[df['item'] == item]['price'].mean()
            price_data.append({
                'Item': item,
                'Current Price (EGP)': current_price,
                'Optimal Price (EGP)': best_price,
                'Projected Profit Increase (EGP)': increase
            })
        price_df = pd.DataFrame(price_data)
        st.dataframe(price_df)

    with st.expander("📈 Visualizations"):
        hourly_profit = df.groupby('hour')['profit'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(hourly_profit['hour'], hourly_profit['profit'])
        ax.set_xlabel('Hour')
        ax.set_ylabel('Total Profit (EGP)')
        ax.set_title('Profit by Hour')
        st.pyplot(fig)

        top_items = df.groupby('item')['profit'].sum().nlargest(10).reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.barh(top_items['item'], top_items['profit'])
        ax2.set_xlabel('Total Profit (EGP)')
        ax2.set_title('Top 10 Items by Profit')
        st.pyplot(fig2)
else:
    st.info("Please select a data source or upload a CSV file.")
