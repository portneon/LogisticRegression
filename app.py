import streamlit as st
import numpy as np
import plotly.graph_objects as plotly_go
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==========================================
# CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(page_title="Interactive Linear Regression Visualization", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

st.title("Interactive Visualization of Linear Regression")
st.markdown("Discover how a model fits data, minimizes error, and converges to an optimal solution.")

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'gd_m' not in st.session_state:
    st.session_state.gd_m = 0.0
if 'gd_b' not in st.session_state:
    st.session_state.gd_b = 0.0
if 'run_gd' not in st.session_state:
    st.session_state.run_gd = False
if 'gd_history' not in st.session_state:
    st.session_state.gd_history = []

# ==========================================
# DATA GENERATION
# ==========================================
def generate_data(n_samples=50, noise_level=5.0, add_outliers=False):
    np.random.seed(42)
    X = np.random.uniform(-10, 10, n_samples)
    y_true = 3 * X + 10  # True parameters: m=3, b=10
    
    # Add noise
    y = y_true + np.random.normal(0, noise_level, n_samples)
    
    # Add outliers
    if add_outliers:
        outlier_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y[outlier_indices] += np.random.choice([-40, 40], size=len(outlier_indices))
        
    return X, y, y_true

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("🎛️ Controls")
    
    st.subheader("Data Settings")
    noise_level = st.slider("Noise Level", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
    add_outliers = st.toggle("Add Outliers", value=False)
    
    st.markdown("---")
    
    st.subheader("Hypothesis Parameters")
    st.markdown("Adjust manually to minimize the error!")
    m_slider = st.slider("Slope (m)", min_value=-10.0, max_value=15.0, value=0.0, step=0.1)
    b_slider = st.slider("Intercept (b)", min_value=-20.0, max_value=30.0, value=0.0, step=0.5)
    
    st.markdown("---")
    
    st.subheader("Gradient Descent")
    learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    epochs = st.slider("Iterations", 10, 500, 100, 10)
    
    if st.button("Run Gradient Descent", type="primary"):
        st.session_state.run_gd = True
    if st.button("Reset Parameters"):
        st.session_state.run_gd = False
        st.session_state.gd_m = 0.0
        st.session_state.gd_b = 0.0
        st.session_state.gd_history = []
        # Need to force a rerun, streamlit trick: do nothing and it reruns normally or use experimental feature

# Fetch Data
X, y, y_true = generate_data(50, noise_level, add_outliers)

# Use manual sliders if GD is not running, otherwise use GD state
m = st.session_state.gd_m if st.session_state.run_gd else m_slider
b = st.session_state.gd_b if st.session_state.run_gd else b_slider

# Compute hypothesis
y_pred = m * X + b
mse = np.mean((y - y_pred)**2)

# Fit Scikit-Learn Optimal Model for reference
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
m_opt = model.coef_[0]
b_opt = model.intercept_

# ==========================================
# MAIN TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Data & Line Fit", 
    "📏 Error (MSE)", 
    "🏔️ Loss Landscape",
    "🏃‍♂️ Gradient Descent",
    "📉 Learning Rate",
    "🌪️ Noise & Robustness"
])

def create_base_scatter(X, y):
    fig = plotly_go.Figure()
    fig.add_trace(plotly_go.Scatter(x=X, y=y, mode='markers', name='Data Elements', marker=dict(color='blue', size=8, opacity=0.6)))
    # Add Optimal Line
    x_range = np.array([-11, 11])
    y_opt_range = m_opt * x_range + b_opt
    fig.add_trace(plotly_go.Scatter(x=x_range, y=y_opt_range, mode='lines', name='Optimal Least Squares Line', line=dict(color='green', dash='dot')))
    return fig

# -----------------
# TAB 1: Data & Line Fit
# -----------------
with tab1:
    st.header("Data Distribution & Line Fitting")
    st.markdown("Linear Regression assumes a linear relationship between input $X$ and output $y$, approximated by a line: **$y = mx + b$**. Your goal is to find the best $m$ and $b$.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig1 = create_base_scatter(X, y)
        x_range = np.array([-11, 11])
        y_pred_range = m * x_range + b
        fig1.add_trace(plotly_go.Scatter(x=x_range, y=y_pred_range, mode='lines', name='Hypothesis (y = mx + b)', line=dict(color='red', width=3)))
        fig1.update_layout(xaxis_title="X", yaxis_title="y", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.metric(label="Current Slope (m)", value=f"{m:.2f}")
        st.metric(label="Current Intercept (b)", value=f"{b:.2f}")
        st.metric(label="Optimal Slope (m)", value=f"{m_opt:.2f}")
        st.metric(label="Optimal Intercept (b)", value=f"{b_opt:.2f}")

# -----------------
# TAB 2: Error Visualization (MSE)
# -----------------
with tab2:
    st.header("Error / Loss Function (MSE)")
    st.markdown("Model performance is measured using Mean Squared Error (MSE). Notice how the residual lines (errors) change as you move the sliders.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig2 = create_base_scatter(X, y)
        fig2.add_trace(plotly_go.Scatter(x=x_range, y=y_pred_range, mode='lines', name='Hypothesis Line', line=dict(color='red', width=3)))
        
        # Add residual lines
        for i in range(len(X)):
            fig2.add_trace(plotly_go.Scatter(
                x=[X[i], X[i]], 
                y=[y[i], y_pred[i]], 
                mode='lines', 
                line=dict(color='orange', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        fig2.update_layout(xaxis_title="X", yaxis_title="y", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
        st.markdown(r"$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$")

# -----------------
# TAB 3: Loss Landscape
# -----------------
with tab3:
    st.header("Loss Surface (Parameter Space)")
    st.markdown("The error is a function of the parameters $m$ and $b$: $J(m, b)$. We can plot this to see the optimization 'landscape'. The red dot is your current position.")
    
    # Calculate MSE grid
    m_grid = np.linspace(-10, 15, 50)
    b_grid = np.linspace(-20, 30, 50)
    M, B = np.meshgrid(m_grid, b_grid)
    
    # Compute MSE for each point in the grid
    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = np.mean((y - (M[i, j] * X + B[i, j]))**2)
            
    view_type = st.radio("View Type", ["Contour Plot", "3D Surface"], horizontal=True)
    
    fig3 = plotly_go.Figure()
    
    if view_type == "Contour Plot":
        fig3.add_trace(plotly_go.Contour(z=Z, x=m_grid, y=b_grid, contours=dict(showlabels=True), colorscale='Viridis'))
        fig3.add_trace(plotly_go.Scatter(x=[m], y=[b], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Current (m,b)'))
        fig3.add_trace(plotly_go.Scatter(x=[m_opt], y=[b_opt], mode='markers', marker=dict(color='green', size=10, symbol='x'), name='Optimal Minima'))
        fig3.update_layout(xaxis_title="Slope (m)", yaxis_title="Intercept (b)", height=600)
    else:
        fig3.add_trace(plotly_go.Surface(z=Z, x=m_grid, y=b_grid, colorscale='Viridis', opacity=0.8))
        fig3.add_trace(plotly_go.Scatter3d(x=[m], y=[b], z=[mse], mode='markers', marker=dict(color='red', size=8, symbol='circle'), name='Current (m,b)'))
        fig3.update_layout(scene=dict(xaxis_title='Slope (m)', yaxis_title='Intercept (b)', zaxis_title='MSE Loss'), height=600)
        
    st.plotly_chart(fig3, use_container_width=True)

# -----------------
# TAB 4: Gradient Descent
# -----------------
with tab4:
    st.header("Gradient Descent Optimization Process")
    st.markdown("Gradient Descent iteratively updates parameters to find the bottom of the loss landscape.")
    
    if st.button("▶️ Play Animation / Run GD locally below"):
        # We will compute GD steps and plot the trajectory on the contour
        pass
        
    gd_m = 0.0
    gd_b = 0.0
    gd_trajectory_m = [gd_m]
    gd_trajectory_b = [gd_b]
    gd_losses = []
    
    for _ in range(epochs):
        # Calculate gradients
        y_pred_gd = gd_m * X + gd_b
        error = y_pred_gd - y
        
        grad_m = (2/len(X)) * np.sum(error * X)
        grad_b = (2/len(X)) * np.sum(error)
        
        # Update parameters
        gd_m = gd_m - learning_rate * grad_m
        gd_b = gd_b - learning_rate * grad_b
        
        gd_trajectory_m.append(gd_m)
        gd_trajectory_b.append(gd_b)
        gd_losses.append(np.mean((y - (gd_m * X + gd_b))**2))
        
    fig4 = plotly_go.Figure()
    fig4.add_trace(plotly_go.Contour(z=Z, x=m_grid, y=b_grid, contours=dict(showlabels=True), colorscale='Blues', opacity=0.6))
    fig4.add_trace(plotly_go.Scatter(x=gd_trajectory_m, y=gd_trajectory_b, mode='lines+markers', marker=dict(size=4), line=dict(color='orange', width=2), name='Trajectory'))
    fig4.add_trace(plotly_go.Scatter(x=[gd_trajectory_m[0]], y=[gd_trajectory_b[0]], mode='markers', marker=dict(color='red', size=10), name='Start'))
    fig4.add_trace(plotly_go.Scatter(x=[gd_trajectory_m[-1]], y=[gd_trajectory_b[-1]], mode='markers', marker=dict(color='green', size=10, symbol='star'), name='End'))
    fig4.update_layout(xaxis_title="Slope (m)", yaxis_title="Intercept (b)", height=600)
    
    st.plotly_chart(fig4, use_container_width=True)
    
    if st.session_state.run_gd:
        st.session_state.gd_m = gd_m
        st.session_state.gd_b = gd_b
        # To make it fully reactive, you could animate here.

# -----------------
# TAB 5: Learning Rate Experiments
# -----------------
with tab5:
    st.header("Learning Rate & Convergence Behavior")
    st.markdown("Compare how different learning rates affect convergence. If it's too high, it diverges; if too low, it's slow.")
    
    lrs_to_test = [0.001, 0.01, 0.05, 0.1]
    
    fig5 = plotly_go.Figure()
    
    for lr in lrs_to_test:
        temp_m = 0.0
        temp_b = 0.0
        losses = []
        for _ in range(epochs):
            y_pred_temp = temp_m * X + temp_b
            err = y_pred_temp - y
            temp_m = temp_m - lr * (2/len(X)) * np.sum(err * X)
            temp_b = temp_b - lr * (2/len(X)) * np.sum(err)
            
            # Clip error to avoid inf in plot from divergence
            l = np.mean((y - (temp_m * X + temp_b))**2)
            if l > 10000:
                l = 10000
            losses.append(l)
            
        fig5.add_trace(plotly_go.Scatter(y=losses, mode='lines', name=f'LR = {lr}'))
        
    fig5.update_layout(xaxis_title="Iteration (Epoch)", yaxis_title="MSE Loss", height=500, yaxis_type="log")
    st.plotly_chart(fig5, use_container_width=True)

# -----------------
# TAB 6: Noise & Robustness
# -----------------
with tab6:
    st.header("Effect of Noise & Outliers")
    st.markdown("Notice how outliers 'pull' the Least Squares line toward them because squared errors disproportionately penalize large deviations.")
    
    # Generate clean data
    X_c, y_c, _ = generate_data(50, 2.0, False)
    # Generate dataset with outliers
    X_o, y_o, _ = generate_data(50, 2.0, True)
    
    # Fit models
    m_clean = LinearRegression().fit(X_c.reshape(-1,1), y_c)
    m_outlier = LinearRegression().fit(X_o.reshape(-1,1), y_o)
    
    fig6 = plotly_go.Figure()
    # Clean data
    fig6.add_trace(plotly_go.Scatter(x=X_c, y=y_c, mode='markers', name='Clean Data Points', marker=dict(color='blue', opacity=0.5)))
    # Outliers (diff between sets)
    fig6.add_trace(plotly_go.Scatter(x=X_o, y=y_o, mode='markers', name='Data With Outliers', marker=dict(color='red', symbol='cross')))
    
    x_range = np.array([-11, 11])
    fig6.add_trace(plotly_go.Scatter(x=x_range, y=m_clean.predict(x_range.reshape(-1,1)), mode='lines', name='Fit on Clean Data', line=dict(color='blue', width=2)))
    fig6.add_trace(plotly_go.Scatter(x=x_range, y=m_outlier.predict(x_range.reshape(-1,1)), mode='lines', name='Fit including Outliers', line=dict(color='red', dash='dash', width=2)))
    
    fig6.update_layout(xaxis_title="X", yaxis_title="y", height=500)
    st.plotly_chart(fig6, use_container_width=True)