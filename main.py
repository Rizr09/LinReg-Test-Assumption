import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up page config
st.set_page_config(
    page_title="Linear Regression Assumptions Dashboard",
    page_icon="📊",
    layout="wide",
)

# Shadcn theme settings and custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton > button { 
        background-color: #4CAF50; 
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .test-definition {
        background-color: #f0f0f0;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .interpretation {
        background-color: #e6f3ff;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title and description
st.title("📊 Enhanced Linear Regression Assumptions Dashboard")
st.write("""
This dashboard allows you to test and visualize the assumptions of a linear regression model. 
Upload one or multiple datasets, select the dependent variable (target), and inspect the assumptions.
""")

# Sidebar for dataset uploads and options
st.sidebar.header("Upload Your Datasets")
uploaded_files = st.sidebar.file_uploader("Upload CSV files", accept_multiple_files=True, type="csv")

# Function to display test definition, formula, and interpretation
def show_test_info(title, definition, formula, interpretation):
    with st.expander(f"ℹ️ {title} Info"):
        st.markdown(f"**Definition:** {definition}")
        st.markdown(f"**Formula:**")
        st.latex(formula)
        st.markdown("**Interpretation:**")
        st.markdown(f'<div class="interpretation">{interpretation}</div>', unsafe_allow_html=True)

# Function to perform linear regression and assumption tests
def perform_regression_analysis(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Linearity test
    fig1 = px.scatter(
        x=model.fittedvalues, 
        y=model.resid, 
        labels={'x': 'Fitted values', 'y': 'Residuals'}, 
        title='Residuals vs Fitted Values'
    )
    fig1.add_shape(type="line", x0=min(model.fittedvalues), y0=0, x1=max(model.fittedvalues), y1=0, line=dict(color="red", dash="dash"))
    
    # Independence test
    dw_stat = durbin_watson(model.resid)
    
    # Homoscedasticity test
    _, pval, __, f_pval = het_breuschpagan(model.resid, model.model.exog)
    
    # Normality test
    shapiro_test = stats.shapiro(model.resid)
    
    # Q-Q Plot
    qq_fig = sm.qqplot(model.resid, line="45", fit=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=qq_fig.gca().lines[1].get_xdata(), y=qq_fig.gca().lines[1].get_ydata(), mode="markers", name="Residuals"))
    fig2.add_trace(go.Scatter(x=qq_fig.gca().lines[0].get_xdata(), y=qq_fig.gca().lines[0].get_ydata(), mode="lines", name="45-degree line"))
    fig2.update_layout(title="Q-Q Plot (Normality Test)", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
    
    # Multicollinearity test
    vif_data = pd.DataFrame()
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["Feature"] = X.columns
    
    return model, fig1, dw_stat, pval, shapiro_test, fig2, vif_data

# Main application logic
if uploaded_files:
    # Process uploaded datasets
    datasets = {file.name: pd.read_csv(file) for file in uploaded_files}
    
    # Dataset selection
    selected_dataset = st.sidebar.selectbox("Choose a dataset", options=list(datasets.keys()))
    data = datasets[selected_dataset]
    
    # Show preview of the selected dataset
    st.write(f"### Dataset: {selected_dataset}")
    st.write(data.head())

    # Variable selection
    dependent_var = st.sidebar.selectbox("Select the dependent variable", options=data.columns)
    independent_vars = st.sidebar.multiselect("Select independent variables", options=[col for col in data.columns if col != dependent_var])

    # Additional options
    st.sidebar.subheader("Additional Options")
    standardize = st.sidebar.checkbox("Standardize variables", value=False)
    test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)

    if dependent_var and independent_vars:
        # Prepare the data for linear regression
        X = data[independent_vars]
        y = data[dependent_var]

        # Standardize if selected
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Perform regression analysis
        model, fig1, dw_stat, pval, shapiro_test, fig2, vif_data = perform_regression_analysis(X_train, y_train)

        # Display results
        st.subheader("Model Summary")
        st.text(model.summary())

        # 1. Linearity test
        st.subheader("1. Linearity Test")
        show_test_info(
            "Linearity",
            "The linearity assumption states that there should be a linear relationship between the independent variables and the dependent variable.",
            r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon",
            """
            • Visual inspection of residuals vs. fitted values plot:
            - If points are randomly scattered around the horizontal line at 0, the linearity assumption likely holds.
            - If there's a clear pattern (e.g., U-shape, funnel shape), the linearity assumption may be violated.
            """
        )
        st.plotly_chart(fig1)
        st.info("Interpret the plot: Look for random scatter around the horizontal line at 0.")

        # 2. Independence test
        st.subheader("2. Independence Test (Durbin-Watson Statistic)")
        show_test_info(
            "Durbin-Watson Test",
            "The Durbin-Watson test checks for autocorrelation in the residuals.",
            r"DW = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}",
            """
            • Durbin-Watson statistic ranges from 0 to 4:
            - DW ≈ 2: No autocorrelation
            - DW < 1.5 or DW > 2.5: Possible autocorrelation
            - DW < 1 or DW > 3: Strong evidence of autocorrelation
            """
        )
        st.metric("Durbin-Watson Statistic", f"{dw_stat:.2f}")
        if 1.5 < dw_stat < 2.5:
            st.success("Residuals are likely independent (No significant autocorrelation).")
        elif dw_stat < 1 or dw_stat > 3:
            st.error("Strong evidence of autocorrelation in residuals.")
        else:
            st.warning("Possible autocorrelation in residuals.")

        # 3. Homoscedasticity test
        st.subheader("3. Homoscedasticity Test (Breusch-Pagan)")
        show_test_info(
            "Breusch-Pagan Test",
            "The Breusch-Pagan test checks for heteroscedasticity in the residuals.",
            r"BP = nR^2 \sim \chi^2_{(p-1)}",
            """
            • Interpret the p-value:
            - p > 0.05: Fail to reject null hypothesis, likely homoscedastic
            - p ≤ 0.05: Reject null hypothesis, likely heteroscedastic
            """
        )
        st.metric("Breusch-Pagan p-value", f"{pval:.4f}")
        if pval > 0.05:
            st.success("Residuals likely have constant variance (Homoscedasticity).")
        else:
            st.warning("Residuals may exhibit non-constant variance (Heteroscedasticity).")

        # 4. Normality test
        st.subheader("4. Normality Test (Shapiro-Wilk Test)")
        show_test_info(
            "Shapiro-Wilk Test",
            "The Shapiro-Wilk test checks if the residuals are normally distributed.",
            r"W = \frac{(\sum_{i=1}^n a_i x_{(i)})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}",
            """
            • Interpret the p-value:
            - p > 0.05: Fail to reject null hypothesis, residuals likely normal
            - p ≤ 0.05: Reject null hypothesis, residuals likely not normal
            """
        )
        st.metric("Shapiro-Wilk p-value", f"{shapiro_test.pvalue:.4f}")
        if shapiro_test.pvalue > 0.05:
            st.success("Residuals are likely normally distributed.")
        else:
            st.warning("Residuals may deviate from normality.")

        # Q-Q Plot
        st.plotly_chart(fig2)
        st.info("""
        Q-Q Plot Interpretation:
        • If points roughly follow the 45-degree line, residuals are likely normally distributed.
        • Deviations from the line indicate departures from normality:
          - S-shape: Skewness
          - Banana shape: Kurtosis issues
        """)

        # 5. Multicollinearity test
        st.subheader("5. Multicollinearity Test (Variance Inflation Factor - VIF)")
        show_test_info(
            "Variance Inflation Factor",
            "VIF measures how much the variance of an estimated regression coefficient increases if your predictors are correlated.",
            r"VIF_j = \frac{1}{1 - R_j^2}",
            """
            • Interpret VIF values:
            - VIF < 5: No significant multicollinearity
            - 5 ≤ VIF < 10: Moderate multicollinearity
            - VIF ≥ 10: High multicollinearity
            """
        )
        st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}).background_gradient(subset=["VIF"], cmap="YlOrRd"))
        if all(vif_data["VIF"] < 5):
            st.success("No significant multicollinearity detected (All VIF < 5).")
        elif any(vif_data["VIF"] > 10):
            st.error("High multicollinearity detected (VIF > 10 for some variables).")
        else:
            st.warning("Moderate multicollinearity detected (5 ≤ VIF < 10 for some variables).")

        # Final Conclusion
        st.subheader("📊 Final Conclusion")
        conclusion_items = []
        if 1.5 < dw_stat < 2.5:
            conclusion_items.append("✅ Independence assumption likely holds")
        else:
            conclusion_items.append("❌ Independence assumption may be violated")
        
        if pval > 0.05:
            conclusion_items.append("✅ Homoscedasticity assumption likely holds")
        else:
            conclusion_items.append("❌ Homoscedasticity assumption may be violated")
        
        if shapiro_test.pvalue > 0.05:
            conclusion_items.append("✅ Normality assumption likely holds")
        else:
            conclusion_items.append("❌ Normality assumption may be violated")
        
        if all(vif_data["VIF"] < 5):
            conclusion_items.append("✅ No significant multicollinearity")
        elif any(vif_data["VIF"] > 10):
            conclusion_items.append("❌ High multicollinearity detected")
        else:
            conclusion_items.append("⚠️ Moderate multicollinearity detected")

        for item in conclusion_items:
            st.markdown(f"- {item}")

        if all(item.startswith("✅") for item in conclusion_items):
            st.success("All assumptions appear to be satisfied. The linear regression model is likely appropriate.")
        else:
            st.warning("Some assumptions may be violated. Consider the recommendations below to improve the model.")

        # Recommendations
        st.subheader("🔍 Recommendations")
        if dw_stat <= 1.5 or dw_stat >= 2.5:
            st.info("• Consider using time series analysis techniques or adding lagged variables to address autocorrelation.")
        if pval <= 0.05:
            st.info("• Try variable transformations (e.g., log, square root) or weighted least squares to address heteroscedasticity.")
        if shapiro_test.pvalue <= 0.05:
            st.info("• Investigate outliers and consider robust regression techniques or variable transformations.")
        if any(vif_data["VIF"] > 5):
            st.info("• Consider feature selection, principal component analysis, or ridge regression to address multicollinearity.")

else:
    st.info("Please upload one or more CSV files to begin the analysis.")