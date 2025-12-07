"""
Pizza Intelligence Platform - Streamlit Dashboard
Business Intelligence & Predictive Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page Configuration
st.set_page_config(
    page_title="Pizza Intelligence Platform",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍕 Pizza Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown("**Data-Driven Business Intelligence for Pizza Operations**")

# Load Data
@st.cache_data
def load_data():
    """Load all analysis results and raw transaction data"""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

    data = {}
    try:
        # Analysis results
        data['association_rules'] = pd.read_csv(os.path.join(base_path, "dashboard_data", "bundle_recommendations.csv"))
        data['clusters'] = pd.read_csv(os.path.join(base_path, "dashboard_data", "customer_segments.csv"))

        # Load raw transaction data for temporal analysis
        raw_data = pd.read_excel(os.path.join(data_path, "Data_Model_-_Pizza_Sales.xlsx"), sheet_name='pizza_sales')
        raw_data['order_date'] = pd.to_datetime(raw_data['order_date'])
        raw_data['order_time'] = pd.to_datetime(raw_data['order_time'], format='%H:%M:%S').dt.time
        raw_data['hour'] = pd.to_datetime(raw_data['order_time'], format='%H:%M:%S').dt.hour
        raw_data['day_of_week'] = raw_data['order_date'].dt.day_name()
        raw_data['week'] = raw_data['order_date'].dt.isocalendar().week
        data['transactions'] = raw_data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Make sure all analysis scripts have been run and data files exist.")
        return None

    return data

# Load ML Models - Phase 1 Production Models
@st.cache_resource
def load_ml_models():
    """Load Phase 1 production ML models from MLflow"""
    import mlflow.sklearn
    import json

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models", "production")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results", "ml_model_tracking", "production_config_phase1.json")

    models = {}
    try:
        # Load production config (optional)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                models['config'] = json.load(f)
        else:
            models['config'] = {'deployment_phase': 1}

        # Load demand forecasting model
        demand_model_path = os.path.join(models_dir, "demand_forecasting")
        try:
            models['demand_model'] = mlflow.sklearn.load_model(demand_model_path)
            models['demand_status'] = 'deployed'
        except Exception as e:
            st.warning(f"Demand model not loaded: {e}")
            models['demand_status'] = 'pending'

        # Load clustering model
        cluster_model_path = os.path.join(models_dir, "clustering")
        try:
            models['cluster_model'] = mlflow.sklearn.load_model(cluster_model_path)
            models['cluster_status'] = 'deployed'
        except Exception as e:
            st.warning(f"Clustering model not loaded: {e}")
            models['cluster_status'] = 'pending'

        # Load clustering config
        cluster_config_path = os.path.join(models_dir, "clustering_config.json")
        if os.path.exists(cluster_config_path):
            with open(cluster_config_path, 'r') as f:
                models['cluster_config'] = json.load(f)
        else:
            models['cluster_config'] = {'metrics': {'silhouette_score': 0.83}}

        # Load association rules
        assoc_rules_path = os.path.join(models_dir, "association_rules.csv")
        assoc_config_path = os.path.join(models_dir, "association_rules_config.json")

        try:
            models['association_rules'] = pd.read_csv(assoc_rules_path)
            with open(assoc_config_path, 'r') as f:
                models['association_config'] = json.load(f)
            models['association_status'] = 'deployed'
        except Exception as e:
            st.warning(f"Association rules not available: {e}")
            models['association_status'] = 'pending'

        # Load revenue prediction model
        revenue_model_path = os.path.join(models_dir, "revenue_prediction")
        revenue_config_path = os.path.join(models_dir, "revenue_config.json")

        try:
            models['revenue_model'] = mlflow.sklearn.load_model(revenue_model_path)
            with open(revenue_config_path, 'r') as f:
                models['revenue_config'] = json.load(f)
            models['revenue_status'] = 'deployed'
        except Exception as e:
            st.warning(f"Revenue prediction model not available: {e}")
            models['revenue_status'] = 'pending'

        return models
    except Exception as e:
        st.warning(f"Phase 1 ML models not available: {e}")
        return None

data = load_data()
models = load_ml_models()

if data is None:
    st.error("Failed to load data. Please ensure all analysis scripts have been run.")
    st.stop()

# Sidebar Navigation (AFTER loading data/models)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Executive Dashboard", "🔮 Demand Forecasting (ML)", "👥 Customer Segments", "🔗 Bundle Recommendations", "📈 Sales Trends", "⏰ Staffing & Peak Hours", "📉 Business Metrics"]
)

# Show ML Models deployment status
if models:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ML Models Status**")

    # Demand Forecasting status
    if models.get('demand_status') == 'deployed':
        st.sidebar.success("✅ Demand Forecasting")
    else:
        st.sidebar.info("⏳ Demand Forecasting (Deploying...)")

    # Clustering status
    if models.get('cluster_status') == 'deployed':
        st.sidebar.success("✅ Customer Clustering")
    else:
        st.sidebar.info("⏳ Customer Clustering (Deploying...)")

    # Association Rules status
    if models.get('association_status') == 'deployed':
        st.sidebar.success(f"✅ Association Rules ({models.get('association_config', {}).get('num_rules', 0)} rules)")
    else:
        st.sidebar.info("⏳ Association Rules (Deploying...)")

    # Revenue Prediction status
    if models.get('revenue_status') == 'deployed':
        st.sidebar.success(f"✅ Revenue Prediction (R²: {models.get('revenue_config', {}).get('metrics', {}).get('test_r2', 0):.2f})")
    else:
        st.sidebar.info("⏳ Revenue Prediction (Deploying...)")

# ============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ============================================================================
if page == "📊 Executive Dashboard":
    st.header("Executive Overview")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_pizzas = len(data['clusters'])
        st.metric("Total Pizza Varieties", total_pizzas)

    with col2:
        total_orders = data['transactions']['order_id'].nunique()
        st.metric("Total Orders Analyzed", f"{total_orders:,}")

    with col3:
        total_revenue = data['transactions']['total_price'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

    with col4:
        num_bundles = len(data['association_rules'])
        st.metric("Strategic Bundles Identified", num_bundles)

    st.markdown("---")

    # Revenue by Pizza
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Revenue Generating Pizzas")
        # Get revenue from transactions
        pizza_revenue = data['transactions'].groupby('pizza_name').agg({
            'total_price': 'sum',
            'quantity': 'sum'
        }).reset_index()
        top_revenue = pizza_revenue.nlargest(10, 'total_price')

        fig = px.bar(
            top_revenue,
            x='total_price',
            y='pizza_name',
            orientation='h',
            title="Revenue by Pizza",
            labels={'total_price': 'Total Revenue ($)', 'pizza_name': 'Pizza'},
            color='total_price',
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Best-Selling Pizzas by Volume")
        # Merge clusters with transactions for quantity data
        top_quantity = pizza_revenue.nlargest(10, 'quantity')[['pizza_name', 'quantity']]
        # Add unit price from clusters
        top_quantity = top_quantity.merge(data['clusters'][['pizza_name', 'unit_price']], on='pizza_name', how='left')

        fig = px.bar(
            top_quantity,
            x='quantity',
            y='pizza_name',
            orientation='h',
            title="Sales Volume by Pizza",
            labels={'quantity': 'Units Sold', 'pizza_name': 'Pizza'},
            color='quantity',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Price Analysis
    st.subheader("Price Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Merge clusters with transaction data for price analysis
        price_analysis = data['clusters'][['pizza_name', 'unit_price']].merge(
            pizza_revenue[['pizza_name', 'quantity', 'total_price']],
            on='pizza_name',
            how='left'
        )
        price_analysis['quantity'] = price_analysis['quantity'].fillna(0)
        price_analysis['total_price'] = price_analysis['total_price'].fillna(0)

        fig = px.scatter(
            price_analysis,
            x='unit_price',
            y='quantity',
            size='total_price',
            color='unit_price',
            hover_name='pizza_name',
            title="Price vs. Sales Volume",
            labels={'unit_price': 'Unit Price ($)', 'quantity': 'Units Sold'},
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Price distribution
        fig = px.histogram(
            data['clusters'],
            x='unit_price',
            nbins=20,
            title="Price Distribution",
            labels={'unit_price': 'Unit Price ($)', 'count': 'Number of Pizzas'},
            color_discrete_sequence=['#FF6B35']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Key Insights
    st.markdown("---")
    st.subheader("📌 Key Business Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        avg_price = data['clusters']['unit_price'].mean()
        st.markdown(f"**Average Pizza Price**")
        st.markdown(f"### ${avg_price:.2f}")
        st.markdown("Optimal pricing range identified for menu strategy")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        # Check for ML association rules first, then fallback to static data
        if models and 'association_rules' in models and len(models['association_rules']) > 0:
            avg_lift = models['association_rules']['lift'].mean()
        elif 'association_rules' in data and len(data['association_rules']) > 0:
            avg_lift = data['association_rules']['lift'].mean()
        else:
            avg_lift = 1.09  # Default fallback value
        st.markdown(f"**Average Bundle Strength**")
        st.markdown(f"### {avg_lift:.2f}x")
        st.markdown("Bundle pairs are purchased together more often than by chance")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        # Get top revenue from transactions
        pizza_revenue = data['transactions'].groupby('pizza_name')['total_price'].sum().reset_index()
        top_revenue_pizza = pizza_revenue.nlargest(1, 'total_price').iloc[0]
        st.markdown(f"**Top Revenue Generator**")
        st.markdown(f"### {top_revenue_pizza['pizza_name'].replace('The ', '')}")
        st.markdown(f"${top_revenue_pizza['total_price']:,.0f} in total sales")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DEMAND FORECASTING (ML)
# ============================================================================
elif page == "🔮 Demand Forecasting (ML)":
    st.header("Pizza Demand Forecasting - Machine Learning")
    st.markdown("**Predict hourly pizza demand** using our Ensemble ML model (R² = 0.69)")

    if not models or models.get('demand_status') != 'deployed':
        st.error("Demand forecasting model not available. Please ensure Phase 1 deployment is complete.")
        st.info("The model files may still be deploying. Please refresh the page in a few minutes.")
    else:
        # Display model info
        demand_config = models.get('config', {}).get('demand_forecasting', {
            'model_type': 'Ensemble',
            'metrics': {'test_r2': 0.6923, 'test_rmse': 4.53, 'test_mae': 3.56},
            'use_case': 'Hourly pizza demand prediction'
        })

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", demand_config['model_type'])
        with col2:
            st.metric("Test R² Score", f"{demand_config['metrics']['test_r2']:.4f}")
        with col3:
            st.metric("Test RMSE", f"{demand_config['metrics']['test_rmse']:.2f} pizzas/hr")

        st.markdown("---")

        # Prediction Interface
        st.subheader("🔮 Predict Hourly Demand")

        col1, col2, col3 = st.columns(3)

        with col1:
            pred_date = st.date_input(
                "Select Date",
                value=pd.Timestamp.now(),
                min_value=pd.Timestamp.now() - pd.Timedelta(days=30),
                max_value=pd.Timestamp.now() + pd.Timedelta(days=30)
            )
            pred_hour = st.slider("Hour of Day", 0, 23, 12)

        with col2:
            # Convert date to pandas Timestamp for proper attribute access
            pred_ts = pd.Timestamp(pred_date)
            day_of_week = pred_ts.dayofweek
            month = pred_ts.month
            day_of_month = pred_ts.day
            week_of_year = pred_ts.isocalendar()[1]
            is_weekend = 1 if day_of_week >= 5 else 0

            st.info(f"**Day:** {pred_date.strftime('%A')}")
            st.info(f"**Month:** {pred_date.strftime('%B')}")
            st.info(f"**Weekend:** {'Yes' if is_weekend else 'No'}")

        with col3:
            # Historical averages for reference
            prev_hour_avg = st.number_input(
                "Previous Hour Pizzas (estimate)",
                min_value=0, max_value=100, value=30,
                help="Estimate from historical data or use 30 as default"
            )
            same_hour_yesterday_avg = st.number_input(
                "Same Hour Yesterday (estimate)",
                min_value=0, max_value=100, value=30,
                help="Estimate from historical data or use 30 as default"
            )

        # Predict button
        if st.button("Predict Demand", type="primary"):
            try:
                # Prepare features (18 features total)
                import numpy as np

                # Cyclic encoding
                hour_sin = np.sin(2 * np.pi * pred_hour / 24)
                hour_cos = np.cos(2 * np.pi * pred_hour / 24)
                day_sin = np.sin(2 * np.pi * day_of_week / 7)
                day_cos = np.cos(2 * np.pi * day_of_week / 7)
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
                day_of_month_cos = np.cos(2 * np.pi * day_of_month / 31)

                # Rolling averages (using estimates)
                rolling_3h_avg = prev_hour_avg  # Simplified
                rolling_24h_avg = same_hour_yesterday_avg  # Simplified

                # Create feature vector
                features = pd.DataFrame([{
                    'hour': pred_hour,
                    'day_of_week': day_of_week,
                    'month': month,
                    'day_of_month': day_of_month,
                    'week_of_year': week_of_year,
                    'is_weekend': is_weekend,
                    'hour_sin': hour_sin,
                    'hour_cos': hour_cos,
                    'day_sin': day_sin,
                    'day_cos': day_cos,
                    'month_sin': month_sin,
                    'month_cos': month_cos,
                    'day_of_month_sin': day_of_month_sin,
                    'day_of_month_cos': day_of_month_cos,
                    'prev_hour_pizzas': prev_hour_avg,
                    'same_hour_yesterday': same_hour_yesterday_avg,
                    'rolling_3h_avg': rolling_3h_avg,
                    'rolling_24h_avg': rolling_24h_avg
                }])

                # Make prediction
                prediction = models['demand_model'].predict(features)[0]

                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Predicted Demand",
                        f"{prediction:.1f} pizzas/hour",
                        help="Ensemble model prediction"
                    )

                with col2:
                    # Calculate staff needed (rough estimate)
                    staff_needed = max(3, min(10, int(prediction / 8) + 2))
                    st.metric(
                        "Staff Needed",
                        f"{staff_needed} employees",
                        help="Rough estimate based on demand"
                    )

                with col3:
                    # Inventory recommendation
                    inventory = int(prediction * 1.2)  # 20% buffer
                    st.metric(
                        "Inventory Prep",
                        f"{inventory} pizzas",
                        help="Recommended inventory (20% buffer)"
                    )

                with col4:
                    # Revenue estimate
                    avg_price = 15  # Approximate average pizza price
                    revenue_est = prediction * avg_price
                    st.metric(
                        "Revenue Estimate",
                        f"${revenue_est:.0f}",
                        help="Estimated revenue for this hour"
                    )

                # Contextual insights
                st.markdown("---")
                st.subheader("💡 Insights & Recommendations")

                # Determine demand level
                if prediction > 40:
                    demand_level = "🔥 High Demand"
                    color = "red"
                    recommendations = [
                        "Schedule maximum staff for this hour",
                        "Pre-prepare dough and toppings",
                        "Consider running promotions to smooth demand",
                        "Ensure all ovens are operational"
                    ]
                elif prediction > 25:
                    demand_level = "📈 Moderate Demand"
                    color = "orange"
                    recommendations = [
                        "Maintain standard staffing levels",
                        "Monitor inventory closely",
                        "Have backup ingredients ready",
                        "Standard preparation procedures"
                    ]
                else:
                    demand_level = "📉 Low Demand"
                    color = "green"
                    recommendations = [
                        "Reduce staff to minimum levels",
                        "Focus on prep work and cleaning",
                        "Consider running flash promotions",
                        "Good time for staff training"
                    ]

                st.markdown(f"**Demand Level:** <span style='color:{color}; font-size:20px;'>{demand_level}</span>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Operational Recommendations:**")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")

                with col2:
                    st.markdown("**Key Factors:**")
                    st.markdown(f"- **Time:** {pred_date.strftime('%A')} at {pred_hour}:00")
                    st.markdown(f"- **Weekend:** {'Yes (expect higher demand)' if is_weekend else 'No (weekday patterns)'}")
                    st.markdown(f"- **Previous Hour:** {prev_hour_avg} pizzas")
                    st.markdown(f"- **Yesterday Same Hour:** {same_hour_yesterday_avg} pizzas")

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Model Performance
        st.markdown("---")
        st.subheader("📈 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Details:**")
            st.markdown(f"- **Type:** {demand_config['model_type']}")
            st.markdown(f"- **R² Score:** {demand_config['metrics']['test_r2']:.4f} (69% variance explained)")
            st.markdown(f"- **RMSE:** {demand_config['metrics']['test_rmse']:.2f} pizzas/hour")
            st.markdown(f"- **MAE:** {demand_config['metrics']['test_mae']:.2f} pizzas/hour")
            st.markdown(f"- **Use Case:** {demand_config['use_case']}")

        with col2:
            st.markdown("**Input Features (18 total):**")
            st.markdown(f"- Temporal: hour, day_of_week, month, day_of_month, week_of_year, is_weekend")
            st.markdown(f"- Cyclic: hour_sin/cos, day_sin/cos, month_sin/cos, day_of_month_sin/cos")
            st.markdown(f"- Lag: prev_hour_pizzas, same_hour_yesterday")
            st.markdown(f"- Rolling: 3h average, 24h average")

# ============================================================================
# PAGE 4: BUNDLE RECOMMENDATIONS
# ============================================================================
elif page == "🔗 Bundle Recommendations":
    st.header("Strategic Pizza Bundle Recommendations - Machine Learning")
    st.markdown("**ML-powered product recommendations** using Association Rule Mining (Apriori algorithm)")

    # Check if ML association rules are available
    if models and 'association_rules' in models and len(models['association_rules']) > 0:
        # Display ML model info
        assoc_config = models['association_config']

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Algorithm", assoc_config['model_type'])
        with col2:
            st.metric("Total Rules", assoc_config['num_rules'])
        with col3:
            st.metric("Avg Lift", f"{assoc_config['metrics']['avg_lift']:.4f}")
        with col4:
            st.metric("Avg Confidence", f"{assoc_config['metrics']['avg_confidence']:.4f}")

        st.markdown("---")
        st.subheader("Top Bundle Opportunities")

        # Use ML-generated rules
        rules_display = models['association_rules'].copy()
    elif 'association_rules' in data and len(data['association_rules']) > 0:
        st.warning("ML association rules not yet deployed. Showing analysis from historical data.")
        # Parse bundle data and calculate co-purchase frequency
        rules_display = data['association_rules'].copy()
    else:
        st.error("No association rules data available. Please run the association rules generation script.")
        st.stop()

    # Calculate total orders for frequency calculation
    total_orders = data['transactions']['order_id'].nunique() if 'transactions' in data else 10000

    # Check if required columns exist
    if 'support' in rules_display.columns and 'lift' in rules_display.columns:
        rules_display['copurchase_count'] = (rules_display['support'] * total_orders).round(0).astype(int)
        rules_display['copurchase_freq_pct'] = (rules_display['support'] * 100).round(2)
        rules_display['lift_display'] = rules_display['lift'].round(2)
    else:
        st.error("Association rules data is missing required columns (support, lift).")
        st.stop()

    # Display top 10 rules
    top_rules = rules_display.nlargest(10, 'lift')[
        ['antecedents', 'consequents', 'copurchase_count', 'copurchase_freq_pct', 'lift_display']
    ]

    st.dataframe(
        top_rules,
        column_config={
            "antecedents": "If Customer Orders",
            "consequents": "Also Recommend",
            "copurchase_count": st.column_config.NumberColumn("Co-Purchase Count", format="%d"),
            "copurchase_freq_pct": st.column_config.NumberColumn("Co-Purchase Frequency (%)", format="%.2f%%"),
            "lift_display": st.column_config.NumberColumn("Lift Factor", format="%.2fx")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bundle Strength Analysis")
        st.caption("Each point represents a bundle pair. Larger bubbles = more frequent co-purchases. Higher lift = stronger bundle.")

        fig = px.scatter(
            rules_display,
            x='confidence',
            y='lift',
            size='support',
            hover_data=['antecedents', 'consequents'],
            title="Bundle Confidence vs. Lift Strength",
            labels={
                'confidence': 'Confidence (Reliability)',
                'lift': 'Lift (Strength)',
                'support': 'Co-Purchase Frequency',
                'antecedents': 'Pizza A',
                'consequents': 'Recommended Pizza B'
            },
            color='lift',
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Baseline (no correlation)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Bundles by Purchase Likelihood")

        top_10 = rules_display.nlargest(10, 'lift')
        top_10['bundle_name'] = top_10.apply(
            lambda x: f"{x['antecedents'][:20]}... → {x['consequents'][:20]}...", axis=1
        )

        fig = px.bar(
            top_10,
            x='lift_display',
            y='bundle_name',
            orientation='h',
            title="Lift Factor by Bundle",
            labels={'lift_display': 'Purchase Likelihood (x)', 'bundle_name': 'Bundle'},
            color='lift_display',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Data-driven Implementation Recommendations
    st.markdown("---")
    st.subheader("💡 Actionable Implementation Plan")

    # Get top bundles for specific recommendations
    top_3_bundles = rules_display.nlargest(3, 'lift')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📱 Digital Ordering System (Week 1-2)**")
        st.markdown(f"""
        1. **Configure Website Bundle Widget**
           - Display "{top_3_bundles.iloc[0]['consequents']}" when customer adds "{top_3_bundles.iloc[0]['antecedents']}"
           - Appears {top_3_bundles.iloc[0]['copurchase_count']} times historically ({top_3_bundles.iloc[0]['copurchase_freq_pct']:.1f}% of orders)
           - Offer 10% discount for immediate add-to-cart

        2. **Email Campaign Setup**
           - Target customers who previously ordered items from top bundles
           - Send personalized bundle offers every 2 weeks
           - A/B test discount levels (10% vs 15%)

        3. **Mobile App Push Notifications**
           - Trigger bundle suggestion 5 minutes after cart add
           - Show estimated delivery time for complete bundle
        """)

    with col2:
        st.markdown("**🏪 In-Store Operations (Week 3-4)**")
        bundle_1 = f"{top_3_bundles.iloc[0]['antecedents']} + {top_3_bundles.iloc[0]['consequents']}"
        bundle_2 = f"{top_3_bundles.iloc[1]['antecedents']} + {top_3_bundles.iloc[1]['consequents']}"

        st.markdown(f"""
        1. **Staff Training Session** (2 hours)
           - Memorize top 3 bundles
           - Practice upsell script: "Customers who order [X] also love [Y]"
           - Role-play scenarios for counter and phone orders

        2. **POS System Updates**
           - Create combo SKUs: "BUNDLE-001", "BUNDLE-002", "BUNDLE-003"
           - Pre-program 12% bundle discount
           - Add quick-access buttons for top combinations

        3. **Physical Store Materials**
           - Print table tents highlighting: "{bundle_1}"
           - Create counter display for "{bundle_2}"
           - Update menu boards with bundle pricing
        """)

# ============================================================================
# PAGE 3: CUSTOMER SEGMENTS
# ============================================================================
elif page == "👥 Customer Segments":
    st.header("Customer Segmentation Analysis")
    st.markdown("Understanding customer groups based on ordering patterns and preferences.")

    # Cluster Overview
    st.subheader("Segment Overview")

    # Merge clusters with transactions to get sales data
    transactions_agg = data['transactions'].groupby('pizza_name').agg({
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index()

    clusters_with_sales = data['clusters'].merge(transactions_agg, on='pizza_name', how='left')
    clusters_with_sales['quantity'] = clusters_with_sales['quantity'].fillna(0)
    clusters_with_sales['total_price'] = clusters_with_sales['total_price'].fillna(0)

    # Create segment summaries with descriptive labels
    segments_kmeans = clusters_with_sales.groupby('kmeans_cluster').agg({
        'pizza_name': 'count',
        'unit_price': 'mean',
        'quantity': 'sum',
        'total_price': 'sum',
        'ingredient_count': 'mean'
    }).round(2)
    segments_kmeans.columns = ['Pizzas', 'Avg Price', 'Total Orders', 'Total Revenue', 'Avg Ingredients']

    # Create descriptive segment names based on characteristics
    def get_segment_name(row):
        price = row['Avg Price']
        orders = row['Total Orders']
        ingredients = row['Avg Ingredients']

        # Determine price tier
        price_tier = "Premium" if price > 18 else "Mid-Range" if price > 14 else "Value"

        # Determine complexity
        complexity = "Gourmet" if ingredients > 6 else "Classic"

        # Determine volume
        if orders > segments_kmeans['Total Orders'].median():
            volume = "High-Volume"
        else:
            volume = "Specialty"

        return f"{price_tier} {complexity} ({volume})"

    segments_kmeans['Segment'] = segments_kmeans.apply(get_segment_name, axis=1)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            segments_kmeans[['Segment', 'Pizzas', 'Avg Price', 'Total Orders']],
            hide_index=True,
            use_container_width=True
        )

    with col2:
        fig = px.scatter(
            segments_kmeans,
            x='Avg Price',
            y='Total Orders',
            size='Total Revenue',
            color='Segment',
            hover_data=['Pizzas', 'Avg Ingredients'],
            title="Customer Segments: Price vs. Order Volume",
            labels={'Avg Price': 'Average Price ($)', 'Total Orders': 'Total Orders'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed Segment Analysis
    st.subheader("Detailed Segment Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        # Revenue by segment
        fig = px.pie(
            segments_kmeans,
            values='Total Revenue',
            names='Segment',
            title="Revenue Distribution by Segment",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Orders by segment
        fig = px.bar(
            segments_kmeans,
            x='Segment',
            y='Total Orders',
            title="Order Volume by Segment",
            color='Total Orders',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # DBSCAN Analysis
    st.subheader("Outlier Detection Analysis (DBSCAN)")

    # Use clusters_with_sales which has quantity data
    dbscan_summary = clusters_with_sales.groupby('dbscan_cluster').agg({
        'pizza_name': lambda x: ', '.join(x[:5]) + ('...' if len(x) > 5 else ''),
        'unit_price': 'mean',
        'quantity': 'sum'
    }).round(2)

    outliers = clusters_with_sales[clusters_with_sales['dbscan_cluster'] == -1]
    core_clusters = clusters_with_sales[clusters_with_sales['dbscan_cluster'] != -1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Core Customer Groups", len(core_clusters['dbscan_cluster'].unique()))

    with col2:
        st.metric("Outlier Pizzas Identified", len(outliers))

    with col3:
        outlier_revenue_loss = outliers['total_price'].sum()
        st.metric("Potential Menu Optimization", f"${outlier_revenue_loss:,.0f}")

    # Show outlier pizzas
    st.markdown("**Underperforming Pizzas (Outliers):**")
    outlier_display = outliers[['pizza_name', 'unit_price', 'quantity', 'total_price']].sort_values('total_price')
    st.dataframe(
        outlier_display,
        column_config={
            "pizza_name": "Pizza Name",
            "unit_price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
            "quantity": st.column_config.NumberColumn("Units Sold", format="%d"),
            "total_price": st.column_config.NumberColumn("Total Revenue", format="$%.2f")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # Segment Strategy Recommendations
    st.subheader("📋 Segment-Specific Marketing Strategies")

    # Identify key segments
    premium_segment = segments_kmeans.nlargest(1, 'Avg Price').iloc[0]
    volume_segment = segments_kmeans.nlargest(1, 'Total Orders').iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ### {premium_segment['Segment']}: Premium Customers
        - **Avg Price:** ${premium_segment['Avg Price']:.2f}
        - **Strategy:** VIP loyalty tier, exclusive menu previews
        - **Tactics:** Premium bundles ($40-50), seasonal specialties
        - **Expected Impact:** High margin, brand positioning
        """)

    with col2:
        st.markdown(f"""
        ### {volume_segment['Segment']}: High-Volume Customers
        - **Total Orders:** {volume_segment['Total Orders']:,.0f}
        - **Strategy:** Family meal deals, weekday specials
        - **Tactics:** Multi-pizza bundles, loyalty rewards
        - **Expected Impact:** Volume growth, repeat purchases
        """)

# ============================================================================
# PAGE 5: SALES TRENDS
# ============================================================================
elif page == "📈 Sales Trends":
    st.header("Sales Trend Analysis & Historical Patterns")
    st.markdown("Analyze historical sales patterns and identify trends.")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_category = st.selectbox(
            "Pizza Category",
            options=["All"] + sorted(data['transactions']['pizza_category'].unique().tolist())
        )

    with col2:
        selected_size = st.selectbox(
            "Pizza Size",
            options=["All"] + sorted(data['transactions']['pizza_size'].unique().tolist())
        )

    with col3:
        # Forecast date selector
        max_date = data['transactions']['order_date'].max()
        forecast_days = st.selectbox(
            "Forecast Period",
            options=["Next 7 Days", "Next 14 Days", "Next 30 Days", "Select Specific Date"],
            index=0
        )

    # Specific date selector (conditional)
    if forecast_days == "Select Specific Date":
        specific_date = st.date_input(
            "Select Forecast Date",
            value=max_date + pd.Timedelta(days=7),
            min_value=max_date + pd.Timedelta(days=1),
            max_value=max_date + pd.Timedelta(days=30)
        )

    # Filter data
    filtered_data = data['transactions'].copy()
    if selected_category != "All":
        filtered_data = filtered_data[filtered_data['pizza_category'] == selected_category]
    if selected_size != "All":
        filtered_data = filtered_data[filtered_data['pizza_size'] == selected_size]

    # Aggregate by date
    daily_sales = filtered_data.groupby('order_date').agg({
        'total_price': 'sum',
        'quantity': 'sum',
        'order_id': 'nunique'
    }).reset_index()

    # Calculate forecast dates and period
    if forecast_days == "Next 7 Days":
        n_days = 7
        forecast_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=7, freq='D')
        forecast_title = "Next 7 Days"
    elif forecast_days == "Next 14 Days":
        n_days = 14
        forecast_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=14, freq='D')
        forecast_title = "Next 14 Days"
    elif forecast_days == "Next 30 Days":
        n_days = 30
        forecast_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_title = "Next 30 Days"
    else:  # Specific date
        forecast_dates = [pd.Timestamp(specific_date)]
        n_days = (specific_date - max_date.date()).days
        forecast_title = f"Selected Date: {specific_date.strftime('%Y-%m-%d')}"

    # Generate forecast
    st.markdown("---")
    st.subheader(f"📊 Revenue Forecast ({forecast_title})")

    # Simple forecast based on recent trends and seasonality
    recent_avg = daily_sales.tail(14)['total_price'].mean()
    recent_std = daily_sales.tail(14)['total_price'].std()
    overall_growth = (daily_sales.tail(7)['total_price'].mean() - daily_sales.head(7)['total_price'].mean()) / daily_sales.head(7)['total_price'].mean()

    # Predict for each day based on day of week seasonality
    predictions = []
    for date in forecast_dates:
        # Get average for this day of week
        dow_avg = filtered_data[filtered_data['order_date'].dt.day_name() == date.day_name()].groupby('order_date')['total_price'].sum().mean()
        # Apply growth trend
        predicted = dow_avg * (1 + overall_growth)
        predictions.append({
            'date': date,
            'predicted_revenue': predicted,
            'predicted_quantity': dow_avg / (recent_avg / daily_sales.tail(14)['quantity'].mean()) * (1 + overall_growth),
            'day_of_week': date.day_name()
        })

    forecast_df = pd.DataFrame(predictions)

    # Display forecast
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predicted Revenue", f"${forecast_df['predicted_revenue'].sum():,.0f}")
    with col2:
        st.metric("Avg Daily Revenue", f"${forecast_df['predicted_revenue'].mean():,.0f}")
    with col3:
        st.metric("Est. Growth Rate", f"{overall_growth*100:+.1f}%")

    # Forecast chart
    fig_forecast = go.Figure()

    # Historical data (last 14 days)
    historical_recent = daily_sales.tail(14)
    fig_forecast.add_trace(go.Scatter(
        x=historical_recent['order_date'],
        y=historical_recent['total_price'],
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))

    # Predicted data
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_revenue'],
        mode='lines+markers',
        name='Predicted Revenue',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))

    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
        y=(forecast_df['predicted_revenue'] + recent_std).tolist() + (forecast_df['predicted_revenue'] - recent_std).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))

    fig_forecast.update_layout(
        title='Revenue Forecast with Historical Comparison',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Detailed forecast table
    if forecast_days != "Select Specific Date":
        st.subheader("Detailed Daily Forecast")
        forecast_display = forecast_df.copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
        forecast_display['predicted_revenue'] = forecast_display['predicted_revenue'].round(2)
        forecast_display['predicted_quantity'] = forecast_display['predicted_quantity'].round(0)
        forecast_display.columns = ['Date', 'Predicted Revenue ($)', 'Predicted Pizzas Sold', 'Day of Week']

        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Sales Trend Chart
    st.subheader("Daily Sales Trend")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Revenue Trend', 'Order Volume Trend'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Revenue trend
    fig.add_trace(
        go.Scatter(
            x=daily_sales['order_date'],
            y=daily_sales['total_price'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color='#2ecc71', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ),
        row=1, col=1
    )

    # Add 7-day moving average
    daily_sales['revenue_ma7'] = daily_sales['total_price'].rolling(window=7, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=daily_sales['order_date'],
            y=daily_sales['revenue_ma7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ),
        row=1, col=1
    )

    # Order volume trend
    fig.add_trace(
        go.Scatter(
            x=daily_sales['order_date'],
            y=daily_sales['quantity'],
            mode='lines',
            name='Daily Orders',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(title_text="Pizzas Sold", row=2, col=1)

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Summary Statistics
    st.markdown("---")
    st.subheader("Forecast Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_daily_revenue = daily_sales['total_price'].mean()
        st.metric("Avg Daily Revenue", f"${avg_daily_revenue:,.0f}")

    with col2:
        avg_daily_orders = daily_sales['quantity'].mean()
        st.metric("Avg Daily Volume", f"{avg_daily_orders:.0f} pizzas")

    with col3:
        # Predict next week (simple moving average)
        last_7_days_revenue = daily_sales.tail(7)['total_price'].mean()
        st.metric("Next Week Forecast", f"${last_7_days_revenue * 7:,.0f}")

    with col4:
        # Growth trend
        first_week = daily_sales.head(7)['total_price'].mean()
        last_week = daily_sales.tail(7)['total_price'].mean()
        growth = ((last_week - first_week) / first_week * 100) if first_week > 0 else 0
        st.metric("Trend", f"{growth:+.1f}%", delta=f"vs first week")

    # Note: Revenue prediction will be added in Phase 2
    st.markdown("---")
    st.info("**Revenue Prediction:** This feature will be available in Phase 2 after training completes. Use the 'Demand Forecasting (ML)' page for ML-powered predictions.")

# ============================================================================
# PAGE 6: STAFFING & PEAK HOURS
# ============================================================================
elif page == "⏰ Staffing & Peak Hours":
    st.header("Predictive Staffing Optimization & Demand Forecasting")
    st.markdown("**Predict future staffing needs** based on demand patterns and historical trends.")

    # Add tabs for Historical vs Predictive
    tab1, tab2 = st.tabs(["📈 Demand Forecast & Predictions", "📊 Historical Analysis"])

    with tab1:
        st.subheader("🔮 Staffing Forecast & Predictions")

        # Date selector for staffing forecast
        col1, col2 = st.columns(2)
        with col1:
            staffing_period = st.selectbox(
                "Forecast Period",
                options=["Next Week", "Next 2 Weeks", "Next Month", "Select Specific Date"],
                key="staffing_period"
            )
        with col2:
            if staffing_period == "Select Specific Date":
                max_trans_date = data['transactions']['order_date'].max()
                staffing_date = st.date_input(
                    "Select Date",
                    value=max_trans_date + pd.Timedelta(days=7),
                    min_value=max_trans_date + pd.Timedelta(days=1),
                    max_value=max_trans_date + pd.Timedelta(days=30),
                    key="staffing_date"
                )

        st.markdown("---")

        # Calculate overall hourly patterns across all weeks
        all_hourly_demand = data['transactions'].groupby(['day_of_week', 'hour']).agg({
            'total_price': 'mean',
            'quantity': 'mean',
            'order_id': 'count'
        }).reset_index()

        # Create day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Calculate trend (last 4 weeks vs previous 4 weeks)
        all_weeks = sorted(data['transactions']['week'].unique())
        if len(all_weeks) >= 8:
            recent_weeks = all_weeks[-4:]
            previous_weeks = all_weeks[-8:-4]

            recent_avg = data['transactions'][data['transactions']['week'].isin(recent_weeks)]['total_price'].sum() / 4
            previous_avg = data['transactions'][data['transactions']['week'].isin(previous_weeks)]['total_price'].sum() / 4
            growth_rate = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
        else:
            growth_rate = 0.05  # Default 5% growth assumption

        # Determine forecast period and scope
        if staffing_period == "Next Week":
            forecast_text = "Next Week (7 Days)"
            forecast_days_count = 7
        elif staffing_period == "Next 2 Weeks":
            forecast_text = "Next 2 Weeks (14 Days)"
            forecast_days_count = 14
        elif staffing_period == "Next Month":
            forecast_text = "Next Month (30 Days)"
            forecast_days_count = 30
        else:
            forecast_text = f"Selected Date: {staffing_date.strftime('%B %d, %Y')}"
            forecast_days_count = 1

        st.info(f"**Forecast Period:** {forecast_text} | **Projected Growth:** {growth_rate*100:+.1f}% | **Days to Forecast:** {forecast_days_count}")

        # Apply growth to historical patterns for prediction
        all_hourly_demand['predicted_revenue'] = all_hourly_demand['total_price'] * (1 + growth_rate)
        all_hourly_demand['predicted_quantity'] = all_hourly_demand['quantity'] * (1 + growth_rate)

        # Define thresholds for staffing
        avg_predicted_revenue = all_hourly_demand['predicted_revenue'].mean()
        high_threshold = avg_predicted_revenue * 1.3
        low_threshold = avg_predicted_revenue * 0.7

        # Calculate predicted staffing needs
        all_hourly_demand['predicted_staff'] = all_hourly_demand['predicted_revenue'].apply(
            lambda x: 8 if x > high_threshold else 3 if x < low_threshold else 5
        )

        # Display forecast summary - scale by forecast period
        col1, col2, col3, col4 = st.columns(4)

        # Scale metrics based on forecast period (all_hourly_demand represents 1 week)
        period_multiplier = forecast_days_count / 7.0

        with col1:
            base_weekly_revenue = all_hourly_demand['predicted_revenue'].sum()
            period_revenue = base_weekly_revenue * period_multiplier
            st.metric(
                f"Predicted {forecast_text.split()[0]}{' ' + forecast_text.split()[1] if len(forecast_text.split()) > 1 else ''} Revenue",
                f"${period_revenue:,.0f}",
                f"{growth_rate*100:+.1f}% vs avg"
            )

        with col2:
            base_staff_hours = all_hourly_demand['predicted_staff'].sum()
            total_staff_hours = base_staff_hours * period_multiplier
            st.metric(
                "Total Staff Hours Needed",
                f"{total_staff_hours:.0f} hrs",
                f"{forecast_text.lower()}"
            )

        with col3:
            labor_cost = total_staff_hours * 15
            st.metric(
                "Predicted Labor Cost",
                f"${labor_cost:,.0f}",
                f"@$15/hr"
            )

        with col4:
            peak_staff_day = all_hourly_demand.groupby('day_of_week')['predicted_staff'].sum().idxmax()
            st.metric(
                "Busiest Predicted Day",
                peak_staff_day,
                "Plan accordingly"
            )

        st.markdown("---")

        # Predicted demand heatmap
        st.subheader("📅 Predicted Hourly Demand Heatmap (Next Week)")

        pivot_predicted = all_hourly_demand.pivot(
            index='day_of_week',
            columns='hour',
            values='predicted_revenue'
        ).fillna(0)
        pivot_predicted = pivot_predicted.reindex(day_order)

        fig_pred_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_predicted.values,
            x=pivot_predicted.columns,
            y=pivot_predicted.index,
            colorscale='YlOrRd',
            text=pivot_predicted.values.round(0),
            texttemplate='$%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Predicted Revenue ($)")
        ))

        fig_pred_heatmap.update_layout(
            title='Predicted Hourly Revenue Heatmap for Next Week',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )

        st.plotly_chart(fig_pred_heatmap, use_container_width=True)

        st.markdown("---")

        # Staffing forecast by day
        st.subheader("👥 Predicted Staffing Requirements by Day")

        daily_staffing = all_hourly_demand.groupby('day_of_week').agg({
            'predicted_staff': 'sum',
            'predicted_revenue': 'sum',
            'predicted_quantity': 'sum'
        }).reset_index()
        daily_staffing = daily_staffing.set_index('day_of_week').reindex(day_order).reset_index()

        fig_staff_forecast = go.Figure()

        fig_staff_forecast.add_trace(go.Bar(
            x=daily_staffing['day_of_week'],
            y=daily_staffing['predicted_staff'],
            name='Staff Hours Needed',
            marker_color='#3498db',
            text=daily_staffing['predicted_staff'].round(0),
            textposition='outside'
        ))

        fig_staff_forecast.update_layout(
            title='Predicted Daily Staffing Requirements (Next Week)',
            xaxis_title='Day of Week',
            yaxis_title='Total Staff Hours',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_staff_forecast, use_container_width=True)

        st.markdown("---")

        # Hourly staffing prediction with trend
        st.subheader("⏰ Hour-by-Hour Staffing Forecast")

        hourly_avg = all_hourly_demand.groupby('hour').agg({
            'predicted_staff': 'mean',
            'predicted_revenue': 'mean',
            'predicted_quantity': 'mean'
        }).reset_index()

        hourly_avg['staff_level'] = hourly_avg['predicted_revenue'].apply(
            lambda x: 'High Staffing' if x > high_threshold
            else 'Low Staffing' if x < low_threshold
            else 'Normal Staffing'
        )

        color_map = {
            'High Staffing': '#e74c3c',
            'Normal Staffing': '#f39c12',
            'Low Staffing': '#2ecc71'
        }

        fig_hourly_pred = px.bar(
            hourly_avg,
            x='hour',
            y='predicted_staff',
            color='staff_level',
            title='Predicted Hourly Staffing Levels',
            labels={'predicted_staff': 'Staff Count', 'hour': 'Hour of Day'},
            color_discrete_map=color_map,
            text='predicted_staff'
        )
        fig_hourly_pred.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_hourly_pred.update_layout(height=400)
        st.plotly_chart(fig_hourly_pred, use_container_width=True)

        # Actionable insights
        st.markdown("---")
        st.subheader("🎯 Predictive Staffing Recommendations")

        peak_predicted_hour = hourly_avg.loc[hourly_avg['predicted_revenue'].idxmax()]
        low_predicted_hour = hourly_avg.loc[hourly_avg['predicted_revenue'].idxmin()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            **🔥 Predicted Peak Period**
            - **Time:** {int(peak_predicted_hour['hour'])}:00-{int(peak_predicted_hour['hour'])+1}:00
            - **Forecast Revenue:** ${peak_predicted_hour['predicted_revenue']:.0f}
            - **Staff Needed:** {int(peak_predicted_hour['predicted_staff'])} employees
            - **Growth:** {growth_rate*100:+.1f}% vs historical avg
            """)

        with col2:
            st.markdown(f"""
            **📉 Predicted Low Period**
            - **Time:** {int(low_predicted_hour['hour'])}:00-{int(low_predicted_hour['hour'])+1}:00
            - **Forecast Revenue:** ${low_predicted_hour['predicted_revenue']:.0f}
            - **Staff Needed:** {int(low_predicted_hour['predicted_staff'])} employees
            - **Action:** Schedule prep, training, inventory
            """)

        with col3:
            max_day_staff = daily_staffing.loc[daily_staffing['predicted_staff'].idxmax()]
            st.markdown(f"""
            **📊 Weekly Budget Forecast**
            - **Total Weekly Hours:** {daily_staffing['predicted_staff'].sum():.0f} hrs
            - **Weekly Labor Cost:** ${daily_staffing['predicted_staff'].sum() * 15:,.0f}
            - **Peak Day:** {max_day_staff['day_of_week']}
            - **Peak Day Hours:** {max_day_staff['predicted_staff']:.0f} hrs
            """)

        # Trend analysis
        st.markdown("---")
        st.subheader("📈 Staffing Trend & Growth Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **📊 Demand Trend Analysis**
            - **Growth Rate:** {growth_rate*100:+.1f}% (recent vs previous weeks)
            - **Trend:** {"📈 Increasing" if growth_rate > 0 else "📉 Decreasing"}
            - **Implication:** {"Hire additional staff" if growth_rate > 0.1 else "Maintain current levels" if growth_rate > -0.1 else "Consider reducing hours"}
            - **Confidence:** {"High" if len(all_weeks) >= 8 else "Medium"}
            """)

        with col2:
            if growth_rate > 0:
                st.success(f"""
                **✅ Hiring Recommendation**
                - Projected to need **{growth_rate*100:.0f}% more staff hours** next week
                - Consider hiring **{int((daily_staffing['predicted_staff'].sum() * growth_rate) / 40)}** additional part-time staff
                - Focus hiring for peak hours: **{int(peak_predicted_hour['hour'])}:00-{int(peak_predicted_hour['hour'])+1}:00**
                - Budget increase: **${(daily_staffing['predicted_staff'].sum() * growth_rate * 15):,.0f}**/week
                """)
            else:
                st.warning(f"""
                **⚠️ Optimization Opportunity**
                - Demand is {"stable" if abs(growth_rate) < 0.05 else "declining"}
                - Current staffing levels are adequate
                - Consider cross-training to improve flexibility
                - Focus on efficiency during peak hours
                """)

    with tab2:
        st.subheader("📊 Historical Demand Analysis")

        # Week selector
        available_weeks = sorted(data['transactions']['week'].unique())
        selected_week = st.selectbox(
            "Select Week to Analyze",
            options=available_weeks,
            index=len(available_weeks)-1  # Default to most recent week
        )

        # Filter data for selected week
        week_data = data['transactions'][data['transactions']['week'] == selected_week].copy()

        st.markdown("---")

        # Hourly demand heatmap
        st.subheader(f"Demand Heatmap - Week {selected_week}")

        # Aggregate hourly sales by day of week
        hourly_demand = week_data.groupby(['day_of_week', 'hour']).agg({
            'total_price': 'sum',
            'quantity': 'sum',
            'order_id': 'nunique'
        }).reset_index()

        # Create pivot for heatmap
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_revenue = hourly_demand.pivot(index='day_of_week', columns='hour', values='total_price').fillna(0)
        pivot_revenue = pivot_revenue.reindex(day_order)

        # Revenue heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_revenue.values,
            x=pivot_revenue.columns,
            y=pivot_revenue.index,
            colorscale='YlOrRd',
            text=pivot_revenue.values.round(0),
            texttemplate='$%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Revenue ($)")
        ))

        fig_heatmap.update_layout(
            title=f'Hourly Revenue by Day of Week (Week {selected_week})',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        # Peak hours analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Peak Hours by Day")

            # Find peak hour for each day
            peak_hours = hourly_demand.loc[hourly_demand.groupby('day_of_week')['total_price'].idxmax()]
            peak_hours = peak_hours.reindex(
                peak_hours['day_of_week'].map(lambda x: day_order.index(x)).sort_values().index
            )

            fig_peak = px.bar(
                peak_hours,
                x='day_of_week',
                y='total_price',
                title='Peak Hour Revenue by Day',
                labels={'total_price': 'Revenue ($)', 'day_of_week': 'Day'},
                color='total_price',
                color_continuous_scale='Oranges',
                text='hour'
            )
            fig_peak.update_traces(texttemplate='Peak: %{text}:00', textposition='outside')
            fig_peak.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_peak, use_container_width=True)

        with col2:
            st.subheader("Average Hourly Demand")

            # Overall hourly pattern (averaged across all days in the week)
            avg_hourly = hourly_demand.groupby('hour').agg({
                'total_price': 'mean',
                'quantity': 'mean',
                'order_id': 'mean'
            }).reset_index()

            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=avg_hourly['hour'],
                y=avg_hourly['total_price'],
                name='Avg Revenue',
                marker_color='lightblue',
                yaxis='y'
            ))
            fig_hourly.add_trace(go.Scatter(
                x=avg_hourly['hour'],
                y=avg_hourly['quantity'],
                name='Avg Pizzas Sold',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))

            fig_hourly.update_layout(
                title='Average Hourly Sales Pattern',
                xaxis_title='Hour of Day',
                yaxis=dict(title='Revenue ($)', side='left'),
                yaxis2=dict(title='Pizzas Sold', overlaying='y', side='right'),
                hovermode='x unified',
                height=350
            )

            st.plotly_chart(fig_hourly, use_container_width=True)

        st.markdown("---")

        # Staffing recommendations
        st.subheader("📋 Historical Staffing Patterns")

        # Calculate staffing levels based on demand
        # Define demand thresholds
        avg_revenue = hourly_demand['total_price'].mean()
        high_threshold = avg_revenue * 1.3
        low_threshold = avg_revenue * 0.7

        # Classify hours into demand levels
        staffing_data = avg_hourly.copy()
        staffing_data['staff_level'] = staffing_data['total_price'].apply(
            lambda x: 'High Staffing' if x > high_threshold
            else 'Low Staffing' if x < low_threshold
            else 'Normal Staffing'
        )

        staffing_data['recommended_staff'] = staffing_data['total_price'].apply(
            lambda x: 8 if x > high_threshold
            else 3 if x < low_threshold
            else 5
        )

        # Staffing level chart
        color_map = {
            'High Staffing': '#e74c3c',
            'Normal Staffing': '#f39c12',
            'Low Staffing': '#2ecc71'
        }

        fig_staffing = px.bar(
            staffing_data,
            x='hour',
            y='recommended_staff',
            color='staff_level',
            title=f'Historical Staff Requirements - Week {selected_week}',
            labels={'recommended_staff': 'Staff Count', 'hour': 'Hour of Day'},
            color_discrete_map=color_map,
            text='recommended_staff'
        )
        fig_staffing.update_traces(textposition='outside')
        fig_staffing.update_layout(height=400)
        st.plotly_chart(fig_staffing, use_container_width=True)

        # Detailed recommendations
        st.markdown("---")
        st.subheader(f"📅 Week {selected_week} Insights")

        col1, col2, col3 = st.columns(3)

        # Peak hours
        peak_hour = staffing_data.loc[staffing_data['total_price'].idxmax()]
        low_hour = staffing_data.loc[staffing_data['total_price'].idxmin()]
        total_staff_needed = staffing_data['recommended_staff'].sum()

        with col1:
            st.markdown(f"""
            **🔥 Peak Demand Period**
            - **Hour:** {int(peak_hour['hour'])}:00 - {int(peak_hour['hour'])+1}:00
            - **Avg Revenue:** ${peak_hour['total_price']:.0f}
            - **Staff Needed:** {int(peak_hour['recommended_staff'])} employees
            - **Action:** Schedule experienced staff, prep stations fully stocked
            """)

        with col2:
            st.markdown(f"""
            **📉 Low Demand Period**
            - **Hour:** {int(low_hour['hour'])}:00 - {int(low_hour['hour'])+1}:00
            - **Avg Revenue:** ${low_hour['total_price']:.0f}
            - **Staff Needed:** {int(low_hour['recommended_staff'])} employees
            - **Action:** Utilize for prep work, cleaning, inventory checks
            """)

        with col3:
            st.markdown(f"""
            **📊 Daily Staffing Budget**
            - **Total Daily Hours:** {total_staff_needed} staff-hours
            - **Avg Hourly Cost:** $15/hour
            - **Daily Labor Cost:** ${total_staff_needed * 15:.0f}
            - **Action:** Adjust schedules weekly based on demand patterns
            """)

        # Day-specific insights
        st.markdown("---")
        st.subheader("📅 Day-Specific Insights")

        # Find busiest and slowest days
        daily_totals = week_data.groupby('day_of_week').agg({
            'total_price': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        daily_totals = daily_totals.set_index('day_of_week').reindex(day_order).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            busiest_day = daily_totals.loc[daily_totals['total_price'].idxmax()]
            st.info(f"""
            **🚀 Busiest Day: {busiest_day['day_of_week']}**
            - Revenue: ${busiest_day['total_price']:,.0f}
            - Orders: {int(busiest_day['order_id'])}
            - Recommendation: Full staff, extra prep, extend hours if possible
            """)

        with col2:
            slowest_day = daily_totals.loc[daily_totals['total_price'].idxmin()]
            st.warning(f"""
            **📉 Slowest Day: {slowest_day['day_of_week']}**
            - Revenue: ${slowest_day['total_price']:,.0f}
            - Orders: {int(slowest_day['order_id'])}
            - Recommendation: Minimal staff, run promotions, schedule training
            """)

# ============================================================================
# PAGE 7: BUSINESS METRICS
# ============================================================================
elif page == "📉 Business Metrics":
    st.header("Key Business Performance Metrics")

    # Calculate metrics from transactions data
    total_revenue = data['transactions']['total_price'].sum()
    total_quantity = data['transactions']['quantity'].sum()
    total_orders = data['transactions']['order_id'].nunique()
    avg_order_value = total_revenue / total_orders
    num_pizzas = len(data['clusters'])

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

    with col2:
        st.metric("Total Orders", f"{total_orders:,}")

    with col3:
        st.metric("Average Order Value", f"${avg_order_value:.2f}")

    with col4:
        st.metric("Menu Items", num_pizzas)

    st.markdown("---")

    # Revenue Analysis
    st.subheader("Revenue Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Pareto analysis - use transaction data
        pizza_revenue_analysis = data['transactions'].groupby('pizza_name').agg({
            'total_price': 'sum',
            'quantity': 'sum'
        }).reset_index()

        sorted_pizzas = pizza_revenue_analysis.sort_values('total_price', ascending=False).copy()
        sorted_pizzas['cumulative_revenue'] = sorted_pizzas['total_price'].cumsum()
        sorted_pizzas['cumulative_pct'] = (sorted_pizzas['cumulative_revenue'] / total_revenue * 100)
        sorted_pizzas['rank'] = range(1, len(sorted_pizzas) + 1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_pizzas['rank'],
            y=sorted_pizzas['total_price'],
            name='Revenue',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=sorted_pizzas['rank'],
            y=sorted_pizzas['cumulative_pct'],
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='red', width=3)
        ))

        fig.update_layout(
            title='Pareto Analysis: Revenue Concentration',
            xaxis_title='Pizza Rank',
            yaxis=dict(title='Revenue ($)'),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Find 80/20
        top_80_count = len(sorted_pizzas[sorted_pizzas['cumulative_pct'] <= 80])
        st.info(f"📊 **80/20 Rule:** Top {top_80_count} pizzas ({top_80_count/num_pizzas*100:.0f}%) generate 80% of revenue")

    with col2:
        # Revenue by category
        category_map = {0: 'Chicken', 1: 'Meat', 2: 'Vegetarian', 3: 'Specialty'}

        # Merge transaction data with clusters to get category info
        category_revenue_data = pizza_revenue_analysis.merge(
            data['clusters'][['pizza_name', 'category_encoded']],
            on='pizza_name',
            how='left'
        )
        category_revenue = category_revenue_data.groupby('category_encoded')['total_price'].sum().reset_index()
        category_revenue['category'] = category_revenue['category_encoded'].map(category_map)

        fig = px.pie(
            category_revenue,
            values='total_price',
            names='category',
            title='Revenue by Category',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Performance Matrix
    st.subheader("Pizza Performance Matrix")

    # Merge clusters with transaction data for performance analysis
    performance_data = data['clusters'][['pizza_name', 'unit_price']].merge(
        pizza_revenue_analysis[['pizza_name', 'quantity', 'total_price']],
        on='pizza_name',
        how='left'
    )
    performance_data['quantity'] = performance_data['quantity'].fillna(0)
    performance_data['total_price'] = performance_data['total_price'].fillna(0)

    # Calculate quadrants
    median_price = performance_data['unit_price'].median()
    median_quantity = performance_data['quantity'].median()

    performance_data['performance'] = performance_data.apply(
        lambda x: 'Premium Star' if x['unit_price'] > median_price and x['quantity'] > median_quantity
        else 'Value Star' if x['unit_price'] <= median_price and x['quantity'] > median_quantity
        else 'Premium Niche' if x['unit_price'] > median_price and x['quantity'] <= median_quantity
        else 'Underperformer',
        axis=1
    )

    fig = px.scatter(
        performance_data,
        x='unit_price',
        y='quantity',
        size='total_price',
        color='performance',
        hover_name='pizza_name',
        title="Performance Matrix: Price vs. Sales Volume",
        labels={'unit_price': 'Unit Price ($)', 'quantity': 'Units Sold'},
        color_discrete_map={
            'Premium Star': '#2ecc71',
            'Value Star': '#3498db',
            'Premium Niche': '#f39c12',
            'Underperformer': '#e74c3c'
        }
    )

    fig.add_hline(y=median_quantity, line_dash="dash", line_color="gray", annotation_text="Median Volume")
    fig.add_vline(x=median_price, line_dash="dash", line_color="gray", annotation_text="Median Price")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Performance summary
    col1, col2, col3, col4 = st.columns(4)

    perf_counts = performance_data['performance'].value_counts()

    with col1:
        st.metric("Premium Stars", perf_counts.get('Premium Star', 0), help="High price, high volume")

    with col2:
        st.metric("Value Stars", perf_counts.get('Value Star', 0), help="Low price, high volume")

    with col3:
        st.metric("Premium Niche", perf_counts.get('Premium Niche', 0), help="High price, low volume")

    with col4:
        st.metric("Underperformers", perf_counts.get('Underperformer', 0), help="Low price, low volume")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p><strong>Pizza Intelligence Platform</strong></p>
        <p>Powered by Machine Learning | MLflow Tracking | Databricks Community Edition</p>
        <p>🍕 Data-Driven Decisions for Better Business Outcomes</p>
    </div>
""", unsafe_allow_html=True)
