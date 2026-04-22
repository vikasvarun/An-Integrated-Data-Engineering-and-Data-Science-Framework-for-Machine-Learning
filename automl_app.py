"""
Professional AutoML Platform v2.0
Advanced Machine Learning System with Interactive Model Selection
Built with PyCaret v3 and Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AutoML Platform Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - PROFESSIONAL DESIGN
# ==========================================
st.markdown("""
    <style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #16213e;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        border-bottom: 3px solid #0f3460;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #0f3460;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    /* Cards and Containers */
    .stCard {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Data Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
    }
    
    /* Success */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    /* Model Card */
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-left: 4px solid #764ba2;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: #f4f4f4;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #e74c3c;
    }
    
    /* Upload Area */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Checkbox */
    .stCheckbox {
        background: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'Upload'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'setup_df' not in st.session_state:
    st.session_state.setup_df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'all_models' not in st.session_state:
    st.session_state.all_models = None
if 'selected_models_for_ensemble' not in st.session_state:
    st.session_state.selected_models_for_ensemble = []
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 2rem; margin: 0;'>🤖 AutoML</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Professional Edition</p>
    </div>
""", unsafe_allow_html=True)

pages = [
    "📤 Upload Dataset",
    "🎯 Column Selection",
    "🔧 Preprocessing & EDA",
    "🚀 Train Models",
    "📊 Model Results",
    "🧪 Testing & Deployment"
]

# Map internal page names to display names
page_map = {
    "Upload": "📤 Upload Dataset",
    "Selection": "🎯 Column Selection",
    "Preprocessing": "🔧 Preprocessing & EDA",
    "Modeling": "🚀 Train Models",
    "Results": "📊 Model Results",
    "Testing": "🧪 Testing & Deployment"
}

# Find current page index
current_display = page_map.get(st.session_state.page, "📤 Upload Dataset")
try:
    current_index = pages.index(current_display)
except ValueError:
    current_index = 0

page = st.sidebar.radio("Navigation", pages, index=current_index)

# Update session state page based on sidebar selection
page_names = ["Upload", "Selection", "Preprocessing", "Modeling", "Results", "Testing"]
for idx, p in enumerate(pages):
    if p == page:
        st.session_state.page = page_names[idx]
        break

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white;'>
    <h4 style='margin: 0 0 0.5rem 0; color: white;'>📚 Features</h4>
    <ul style='margin: 0; padding-left: 1.2rem; font-size: 0.85rem; line-height: 1.8;'>
        <li>🎯 Smart Task Detection</li>
        <li>🔧 Auto Preprocessing</li>
        <li>🤖 20+ ML Algorithms</li>
        <li>📊 Interactive Visualizations</li>
        <li>🎨 Model Selection UI</li>
        <li>🔗 Ensemble Methods</li>
        <li>🚀 Easy Deployment</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); font-size: 0.75rem; padding: 1rem;'>
    <p style='margin: 0;'>AutoML Platform v2.0</p>
    <p style='margin: 0.25rem 0 0 0;'>Powered by PyCaret & Streamlit</p>
</div>
""", unsafe_allow_html=True)

print(f"Current page content will follow for: {st.session_state.page}")

# ==========================================
# PAGE 1: UPLOAD DATASET
# ==========================================
if st.session_state.page == 'Upload':
    st.markdown("<h1>📤 Dataset Upload</h1>", unsafe_allow_html=True)
    st.markdown("### Upload your CSV file to begin your machine learning journey")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file ",
            type=['csv'],
            help="Upload a CSV file containing your dataset"
        )
        
        if uploaded_file is not None:
            try:
                file_size = uploaded_file.size / (1024 * 1024)
                
                if file_size > 10:
                    st.error("❌ File size exceeds 10MB. Please upload a smaller file.")
                else:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    
                    st.success(f"✅ File uploaded successfully! Size: {file_size:.2f} MB")
                    
                    # Dataset Preview
                    st.markdown("### 📋 Dataset Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Dataset Information
                    st.markdown("### 📊 Dataset Information")
                    
                    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                    
                    with info_col1:
                        st.metric("Total Rows", f"{df.shape[0]:,}")
                    with info_col2:
                        st.metric("Total Columns", df.shape[1])
                    with info_col3:
                        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
                    with info_col4:
                        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
                    
                    # Column Details Table
                    st.markdown("### 🔍 Column Details")
                    col_info = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': df.dtypes.values,
                        'Missing': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
                        'Unique': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                    # Summary Statistics
                    st.markdown("### 📈 Summary Statistics")
                    st.dataframe(df.describe(), use_container_width=True)
                    
                    # Next button
                    if st.button("➡️ Proceed to Column Selection", type="primary"):
                        st.session_state.page = 'Selection'
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class='stCard'>
            <h3 style='margin-top: 0;'>💡 Tips</h3>
            <p><strong>Data Requirements:</strong></p>
            <ul>
                <li>CSV format only</li>
                <li>Max size: 10MB</li>
                <li>Clean column names</li>
                <li>No special characters</li>
            </ul>
            <p><strong>Best Practices:</strong></p>
            <ul>
                <li>Remove duplicates</li>
                <li>Handle missing values</li>
                <li>Check data types</li>
                <li>Verify target column</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: COLUMN SELECTION
# ==========================================
elif st.session_state.page == 'Selection':
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first!")
        if st.button("⬅️ Back to Upload"):
            st.session_state.page = 'Upload'
            st.rerun()
    else:
        st.markdown("<h1>🎯 Column Selection</h1>", unsafe_allow_html=True)
        st.markdown("### Define your machine learning task by selecting target and feature columns")
        
        df = st.session_state.df
        
        st.info("""
        💡 **Important:** Select which column you want to predict (target variable). 
        This automatically determines if this is a Classification or Regression task.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Step 1: Target Selection
            st.markdown("### 🎯 Step 1: Select Target Column")
            st.markdown("**What do you want to predict?**")
            
            target_column = st.selectbox(
                "Target Column (Outcome/Dependent Variable):",
                options=["-- Select a column --"] + df.columns.tolist(),
                help="The variable your model will learn to predict"
            )
            
            if target_column == "-- Select a column --":
                st.warning("⚠️ Please select a target column to continue")
                st.stop()
            
            st.success(f"✅ **Selected:** `{target_column}`")
            st.session_state.target_column = target_column
            
            # Target Information
            st.markdown("#### 📊 Target Column Analysis")
            
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                st.metric("Data Type", str(df[target_column].dtype))
            with t_col2:
                st.metric("Unique Values", df[target_column].nunique())
            with t_col3:
                st.metric("Missing", df[target_column].isnull().sum())
            with t_col4:
                st.metric("Missing %", f"{(df[target_column].isnull().sum() / len(df) * 100):.1f}%")
            
            # Auto-detect problem type
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 20:
                st.session_state.problem_type = 'classification'
                
                st.markdown("---")
                st.markdown("""
                <div class='success-box'>
                    <h3 style='margin: 0 0 0.5rem 0; color: white;'>🎯 CLASSIFICATION TASK DETECTED</h3>
                    <p style='margin: 0; color: rgba(255,255,255,0.9);'>
                        Predicting categories or classes
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Number of Classes:** {df[target_column].nunique()}")
                
                # Class distribution chart
                st.markdown("**Class Distribution:**")
                class_dist = df[target_column].value_counts()
                
                fig = px.bar(
                    x=class_dist.index.astype(str),
                    y=class_dist.values,
                    labels={'x': target_column, 'y': 'Count'},
                    title=f'Distribution of {target_column}',
                    color=class_dist.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Class balance check
                min_class_count = class_dist.min()
                if min_class_count < 2:
                    st.error("""
                    ⚠️ **Critical:** Some classes have only 1 sample. 
                    Enable "Remove small classes" in Preprocessing step.
                    """)
                elif min_class_count < 10:
                    st.warning(f"""
                    ⚠️ **Warning:** Smallest class has only {min_class_count} samples. 
                    Consider collecting more data for better model performance.
                    """)
            else:
                st.session_state.problem_type = 'regression'
                
                st.markdown("---")
                st.markdown("""
                <div class='success-box'>
                    <h3 style='margin: 0 0 0.5rem 0; color: white;'>📈 REGRESSION TASK DETECTED</h3>
                    <p style='margin: 0; color: rgba(255,255,255,0.9);'>
                        Predicting continuous numerical values
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Distribution plot
                st.markdown("**Target Distribution:**")
                fig = px.histogram(
                    df,
                    x=target_column,
                    nbins=50,
                    title=f'Distribution of {target_column}',
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                stats = df[target_column].describe()
                with stats_col1:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                with stats_col2:
                    st.metric("Median", f"{stats['50%']:.2f}")
                with stats_col3:
                    st.metric("Min", f"{stats['min']:.2f}")
                with stats_col4:
                    st.metric("Max", f"{stats['max']:.2f}")
            
            # Step 2: Feature Selection
            st.markdown("---")
            st.markdown("### 🔧 Step 2: Select Feature Columns")
            st.markdown("**Which columns should be used to make predictions?**")
            
            available_features = [col for col in df.columns if col != target_column]
            
            feature_method = st.radio(
                "Feature selection method:",
                ["✅ Use All Features (Recommended)", "🎯 Select Specific Features"],
                help="Start with all features for best results"
            )
            
            if "All" in feature_method:
                feature_columns = available_features
                st.success(f"✅ All {len(feature_columns)} features selected")
                
                with st.expander("📋 View Selected Features"):
                    st.write(", ".join(feature_columns))
            else:
                feature_columns = st.multiselect(
                    "Choose features:",
                    options=available_features,
                    default=available_features[:min(10, len(available_features))]
                )
                
                if not feature_columns:
                    st.warning("⚠️ Select at least one feature")
                    st.stop()
            
            st.session_state.feature_columns = feature_columns
            
            # Feature summary
            st.markdown("### 📊 Selected Features Summary")
            feature_info = pd.DataFrame({
                'Feature': feature_columns,
                'Type': [str(df[col].dtype) for col in feature_columns],
                'Unique': [df[col].nunique() for col in feature_columns],
                'Missing %': [(df[col].isnull().sum() / len(df) * 100).round(1) for col in feature_columns]
            })
            st.dataframe(feature_info, use_container_width=True)
            
            # Navigation
            st.markdown("---")
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬅️ Back to Upload"):
                    st.session_state.page = 'Upload'
                    st.rerun()
            with nav_col2:
                if st.button("➡️ Proceed to Preprocessing", type="primary"):
                    st.session_state.page = 'Preprocessing'
                    st.rerun()
        
        with col2:
            st.markdown("""
            <div class='stCard'>
                <h3 style='margin-top: 0;'>💡 Quick Guide</h3>
                <p><strong>Target Column Examples:</strong></p>
                <ul>
                    <li>🏥 Medical: Diagnosis, Disease</li>
                    <li>💰 Finance: Price, Profit, Sales</li>
                    <li>📊 Business: Churn, Revenue</li>
                    <li>🎯 General: Outcome, Category</li>
                </ul>
                <p><strong>Feature Columns:</strong></p>
                <p>All other columns used as inputs to predict the target.</p>
                <p><strong>💡 Tip:</strong> Start with all features, then refine based on results!</p>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: PREPROCESSING & EDA
# ==========================================
elif st.session_state.page == 'Preprocessing':
    if st.session_state.df is None or st.session_state.target_column is None:
        st.warning("⚠️ Please complete previous steps first!")
        if st.button("⬅️ Back"):
            st.session_state.page = 'Selection'
            st.rerun()
    else:
        st.markdown("<h1>🔧 Data Preprocessing & Analysis</h1>", unsafe_allow_html=True)
        
        df = st.session_state.df
        target = st.session_state.target_column
        features = st.session_state.feature_columns
        
        working_df = df[[target] + features].copy()
        
        # Preprocessing Options
        st.markdown("### ⚙️ Preprocessing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.checkbox("✅ Impute missing values", value=True)
            encode_categorical = st.checkbox("✅ Encode categorical variables", value=True)
            scale_features = st.checkbox("✅ Normalize/Scale features", value=True)
        
        with col2:
            remove_outliers = st.checkbox("🎯 Remove outliers (IQR)", value=False)
            if st.session_state.problem_type == 'classification':
                remove_small_classes = st.checkbox("🎯 Remove small classes", value=False)
                if remove_small_classes:
                    min_samples = st.slider("Min samples per class:", 2, 20, 5)
        
        if st.button("🔄 Apply Preprocessing", type="primary"):
            with st.spinner("Processing..."):
                processed_df = working_df.copy()
                
                if handle_missing:
                    for col in processed_df.columns:
                        if processed_df[col].isnull().sum() > 0:
                            if processed_df[col].dtype in ['float64', 'int64']:
                                processed_df[col].fillna(processed_df[col].median(), inplace=True)
                            else:
                                processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                
                if remove_outliers:
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col != target or st.session_state.problem_type == 'regression':
                            Q1 = processed_df[col].quantile(0.25)
                            Q3 = processed_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            processed_df = processed_df[
                                (processed_df[col] >= Q1 - 1.5*IQR) & 
                                (processed_df[col] <= Q3 + 1.5*IQR)
                            ]
                
                if st.session_state.problem_type == 'classification' and remove_small_classes:
                    class_counts = processed_df[target].value_counts()
                    classes_to_keep = class_counts[class_counts >= min_samples].index
                    processed_df = processed_df[processed_df[target].isin(classes_to_keep)]
                
                st.session_state.preprocessed_df = processed_df
                st.success("✅ Preprocessing completed!")
        
        # EDA Section
        if st.session_state.preprocessed_df is not None:
            viz_df = st.session_state.preprocessed_df
            
            st.markdown("---")
            st.markdown("### 📊 Exploratory Data Analysis")
            
            tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔥 Correlations", "🎯 Target Analysis"])
            
            with tab1:
                numeric_features = viz_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_features) > 1:
                    selected_feature = st.selectbox("Select feature to visualize:", numeric_features[:10])
                    fig = px.histogram(viz_df, x=selected_feature, color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                numeric_df = viz_df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if st.session_state.problem_type == 'classification':
                    class_dist = viz_df[target].value_counts()
                    fig = px.pie(values=class_dist.values, names=class_dist.index, hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.box(viz_df, y=target, color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Navigation
            st.markdown("---")
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬅️ Back"):
                    st.session_state.page = 'Selection'
                    st.rerun()
            with nav_col2:
                if st.button("➡️ Train Models", type="primary"):
                    st.session_state.page = 'Modeling'
                    st.rerun()


# ==========================================
# PAGE 4: TRAIN MODELS (PROFESSIONAL UI)
# ==========================================
elif st.session_state.page == 'Modeling':
    if st.session_state.preprocessed_df is None and st.session_state.df is None:
        st.warning("⚠️ Please complete previous steps!")
        st.stop()
    
    st.markdown("<h1>🚀 Model Training & Selection</h1>", unsafe_allow_html=True)
    st.markdown("### Professional AutoML with interactive model selection")
    
    # Use preprocessed data if available, otherwise use original
    df = st.session_state.preprocessed_df if st.session_state.preprocessed_df is not None else st.session_state.df
    target = st.session_state.target_column
    problem_type = st.session_state.problem_type
    
    # Training Configuration
    st.markdown("### ⚙️ Training Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        fold_number = st.slider("Cross-Validation Folds:", 2, 10, 5)
    with config_col2:
        train_size = st.slider("Training Set Size:", 0.6, 0.9, 0.7, 0.05)
    with config_col3:
        if problem_type == 'classification':
            sort_metric = st.selectbox("Optimization Metric:", ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall'])
        else:
            sort_metric = st.selectbox("Optimization Metric:", ['R2', 'MAE', 'MSE', 'RMSE'])
    
    # START TRAINING BUTTON
    if not st.session_state.training_complete:
        if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
            try:
                # Import PyCaret
                if problem_type == 'classification':
                    from pycaret.classification import setup, compare_models, pull, finalize_model, save_model
                else:
                    from pycaret.regression import setup, compare_models, pull, finalize_model, save_model
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Setup
                status_text.text("⚙️ Initializing PyCaret environment...")
                progress_bar.progress(20)
                
                try:
                    setup_config = setup(
                        data=df,
                        target=target,
                        fold=fold_number,
                        train_size=train_size,
                        session_id=123,
                        verbose=False,
                        html=False,
                        system_log=False
                    )
                except Exception as e:
                    if "least populated class" in str(e):
                        st.warning("⚠️ Adjusting train_size to 0.7...")
                        setup_config = setup(
                            data=df,
                            target=target,
                            fold=max(2, fold_number-1),
                            train_size=0.7,
                            session_id=123,
                            verbose=False,
                            html=False,
                            system_log=False
                        )
                    else:
                        raise e
                
                st.session_state.setup_df = pull()
                progress_bar.progress(40)
                
                # Compare Models
                status_text.text("🔍 Training and comparing all available models...")
                progress_bar.progress(50)
                
                models = compare_models(
                    sort=sort_metric,
                    n_select=10,  # Get top 10 models
                    verbose=False
                )
                
                comparison = pull()
                st.session_state.comparison_df = comparison
                st.session_state.all_models = models if isinstance(models, list) else [models]
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                st.session_state.training_complete = True
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
                st.exception(e)
    
    # DISPLAY RESULTS AFTER TRAINING
    if st.session_state.training_complete and st.session_state.comparison_df is not None:
        st.success("🎉 Model training completed! Review and select models below.")
        
        # Display Model Comparison
        st.markdown("---")
        st.markdown("### 📊 Model Performance Comparison")
        
        comparison_df = st.session_state.comparison_df
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=[sort_metric], color='lightgreen'), 
                     use_container_width=True)
        
        # Model Selection UI
        st.markdown("---")
        st.markdown("### 🎯 Select Models for Ensemble (Optional)")
        
        st.info("""
        💡 **Ensemble Learning:** Combine multiple models to improve accuracy.
        Select 2-5 models below to create an ensemble using Voting, Stacking, or Blending.
        """)
        
        # Create model cards for selection
        model_names = comparison_df['Model'].tolist()[:10]
        
        selected_models = []
        
        cols = st.columns(2)
        for idx, model_name in enumerate(model_names):
            with cols[idx % 2]:
                model_score = comparison_df[comparison_df['Model'] == model_name][sort_metric].values[0]
                
                # Model card with checkbox
                card_html = f"""
                <div class='model-card'>
                    <h4 style='margin: 0 0 0.5rem 0; color: #667eea;'>{model_name}</h4>
                    <p style='margin: 0; font-size: 1.5rem; font-weight: 700; color: #1a1a2e;'>
                        {sort_metric}: {model_score:.4f}
                    </p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.checkbox(f"Select {model_name}", key=f"model_{idx}"):
                    selected_models.append(idx)
        
        st.session_state.selected_models_for_ensemble = selected_models
        
        # Ensemble Options
        if len(selected_models) >= 2:
            st.markdown("---")
            st.markdown("### 🔗 Create Ensemble Model")
            
            ensemble_method = st.radio(
                "Choose ensemble method:",
                ["Voting", "Stacking", "Blending"],
                help="Voting: Average predictions | Stacking: Train meta-model | Blending: Weighted average"
            )
            
            if st.button("🔗 Create Ensemble", type="primary"):
                try:
                    if problem_type == 'classification':
                        from pycaret.classification import blend_models, stack_models, finalize_model, save_model
                    else:
                        from pycaret.regression import blend_models, stack_models, finalize_model, save_model
                    
                    with st.spinner(f"Creating {ensemble_method} ensemble..."):
                        selected_model_objects = [st.session_state.all_models[i] for i in selected_models]
                        
                        if ensemble_method == "Voting" or ensemble_method == "Blending":
                            ensemble = blend_models(selected_model_objects[:5])
                        else:  # Stacking
                            ensemble = stack_models(selected_model_objects[:5])
                        
                        final_model = finalize_model(ensemble)
                        st.session_state.best_model = final_model
                        
                        model_name = f"ensemble_{ensemble_method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        save_model(final_model, model_name)
                        
                        st.success(f"✅ {ensemble_method} ensemble created and saved!")
                        
                        if st.button("➡️ View Results", type="primary"):
                            st.session_state.page = 'Results'
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Ensemble error: {str(e)}")
        
        # Or use best single model
        st.markdown("---")
        if st.button("✅ Use Best Single Model", type="primary"):
            try:
                if problem_type == 'classification':
                    from pycaret.classification import finalize_model, save_model
                else:
                    from pycaret.regression import finalize_model, save_model
                
                best_model = st.session_state.all_models[0]
                final_model = finalize_model(best_model)
                st.session_state.best_model = final_model
                
                model_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_model(final_model, model_name)
                
                st.success(f"✅ Best model finalized: {model_names[0]}")
                
                if st.button("➡️ View Results", type="primary"):
                    st.session_state.page = 'Results'
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ==========================================
# PAGE 5: MODEL RESULTS
# ==========================================
elif st.session_state.page == 'Results':
    if st.session_state.best_model is None:
        st.warning("⚠️ Please train a model first!")
        if st.button("⬅️ Back to Training"):
            st.session_state.page = 'Modeling'
            st.rerun()
    else:
        st.markdown("<h1>📊 Model Performance & Results</h1>", unsafe_allow_html=True)
        
        problem_type = st.session_state.problem_type
        
        # Import PyCaret
        if problem_type == 'classification':
            from pycaret.classification import plot_model, pull
        else:
            from pycaret.regression import plot_model, pull
        
        model = st.session_state.best_model
        
        # Model Info
        st.markdown("### 🏆 Model Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Model Type", type(model).__name__)
        with info_col2:
            st.metric("Task", problem_type.upper())
        with info_col3:
            st.metric("Status", "✅ Ready")
        
        # Model Comparison
        if st.session_state.comparison_df is not None:
            st.markdown("### 📈 All Models Performance")
            st.dataframe(st.session_state.comparison_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Performance Visualizations")
        
        if problem_type == 'classification':
            viz_tabs = st.tabs(["Confusion Matrix", "AUC Curve", "Feature Importance"])
            
            with viz_tabs[0]:
                try:
                    plot_model(model, plot='confusion_matrix', save=True, verbose=False)
                    st.image('Confusion Matrix.png')
                except:
                    st.info("Visualization not available")
            
            with viz_tabs[1]:
                try:
                    plot_model(model, plot='auc', save=True, verbose=False)
                    st.image('AUC.png')
                except:
                    st.info("Visualization not available")
            
            with viz_tabs[2]:
                try:
                    plot_model(model, plot='feature', save=True, verbose=False)
                    st.image('Feature Importance.png')
                except:
                    st.info("Feature importance not available for this model")
        
        else:  # Regression
            viz_tabs = st.tabs(["Residuals", "Prediction Error", "Feature Importance"])
            
            with viz_tabs[0]:
                try:
                    plot_model(model, plot='residuals', save=True, verbose=False)
                    st.image('Residuals.png')
                except:
                    st.info("Visualization not available")
            
            with viz_tabs[1]:
                try:
                    plot_model(model, plot='error', save=True, verbose=False)
                    st.image('Prediction Error.png')
                except:
                    st.info("Visualization not available")
            
            with viz_tabs[2]:
                try:
                    plot_model(model, plot='feature', save=True, verbose=False)
                    st.image('Feature Importance.png')
                except:
                    st.info("Feature importance not available")
        
        # Navigation
        st.markdown("---")
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬅️ Back to Training"):
                st.session_state.page = 'Modeling'
                st.rerun()
        with nav_col2:
            if st.button("➡️ Test & Deploy", type="primary"):
                st.session_state.page = 'Testing'
                st.rerun()


# ==========================================
# PAGE 6: TESTING & DEPLOYMENT
# ==========================================
elif st.session_state.page == 'Testing':
    if st.session_state.best_model is None:
        st.warning("⚠️ Please train a model first!")
        st.stop()
    
    st.markdown("<h1>🧪 Model Testing & Deployment</h1>", unsafe_allow_html=True)
    
    problem_type = st.session_state.problem_type
    model = st.session_state.best_model
    features = st.session_state.feature_columns
    
    # Import PyCaret
    if problem_type == 'classification':
        from pycaret.classification import predict_model
    else:
        from pycaret.regression import predict_model
    
    st.markdown("### Choose Prediction Method")
    
    prediction_method = st.radio(
        "",
        ["🎯 Single Prediction", "📊 Batch Prediction (CSV)"],
        horizontal=True
    )
    
    if prediction_method == "🎯 Single Prediction":
        st.markdown("### 🎯 Single Instance Prediction")
        st.markdown("Enter values for each feature:")
        
        df = st.session_state.df
        input_data = {}
        
        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                if df[feature].dtype in ['float64', 'int64']:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )
                else:
                    unique_values = df[feature].unique().tolist()
                    input_data[feature] = st.selectbox(feature, options=unique_values)
        
        if st.button("🔮 Make Prediction", type="primary"):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = predict_model(model, data=input_df)
                
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                if problem_type == 'classification':
                    predicted_class = prediction['prediction_label'].values[0]
                    
                    st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='margin: 0; color: white;'>Predicted Class: {predicted_class}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probabilities
                    prob_cols = [col for col in prediction.columns if col.startswith('prediction_score')]
                    if prob_cols:
                        st.markdown("**Prediction Confidence:**")
                        for col in prob_cols:
                            class_name = col.replace('prediction_score_', '')
                            prob = prediction[col].values[0]
                            st.progress(prob, text=f"{class_name}: {prob:.2%}")
                
                else:  # Regression
                    predicted_value = prediction['prediction_label'].values[0]
                    
                    st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='margin: 0; color: white;'>Predicted Value: {predicted_value:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**Input Values:**")
                st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
    
    else:  # Batch Prediction
        st.markdown("### 📊 Batch Prediction")
        
        st.info("Upload a CSV file with the same features as the training data.")
        
        with st.expander("📋 Required Features"):
            st.code(", ".join(features))
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                
                st.success("✅ File uploaded!")
                st.dataframe(test_df.head(), use_container_width=True)
                
                missing_features = set(features) - set(test_df.columns)
                if missing_features:
                    st.error(f"❌ Missing features: {', '.join(missing_features)}")
                else:
                    if st.button("🔮 Make Batch Predictions", type="primary"):
                        with st.spinner("Predicting..."):
                            try:
                                predictions = predict_model(model, data=test_df)
                                
                                st.success("✅ Predictions completed!")
                                
                                st.markdown("### 📊 Results")
                                
                                if problem_type == 'classification':
                                    pred_counts = predictions['prediction_label'].value_counts()
                                    
                                    fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                                               labels={'x': 'Class', 'y': 'Count'},
                                               title='Prediction Distribution')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                st.dataframe(predictions, use_container_width=True)
                                
                                # Download
                                csv = predictions.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Predictions",
                                    csv,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    type="primary"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Navigation
    st.markdown("---")
    if st.button("⬅️ Back to Results"):
        st.session_state.page = 'Results'
        st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; color: rgba(255,255,255,0.6); font-size: 0.75rem;'>
    <p style='margin: 0;'>© 2026 AutoML Platform Pro</p>
    <p style='margin: 0.5rem 0 0 0;'>Powered by PyCaret v3.2</p>
</div>
""", unsafe_allow_html=True)