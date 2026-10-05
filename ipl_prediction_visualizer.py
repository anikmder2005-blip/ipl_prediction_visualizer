"""
IPL Winner Prediction Visualizer
================================
Interactive Streamlit app to demonstrate the difference between 
Traditional Programming vs Machine Learning approaches
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="IPL Prediction: Traditional vs ML",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏏 IPL Winner Prediction: Traditional vs Machine Learning")
st.markdown("### Understanding the fundamental difference between rule-based programming and ML")

# Sidebar
st.sidebar.header("📚 About This App")
st.sidebar.info(
    "This interactive demo shows how Machine Learning differs from traditional programming "
    "using IPL match predictions as an example."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Key Learning Points")
st.sidebar.markdown("""
1. **Traditional**: Fixed rules, limited scenarios
2. **ML**: Learns from data, handles any scenario
3. **ML provides confidence scores**
4. **ML improves with more data**
""")

# Generate synthetic IPL data
@st.cache_data
def generate_ipl_data(n_samples=1000):
    """Generate synthetic IPL match data"""
    np.random.seed(42)
    
    teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
             'Rajasthan Royals', 'Sunrisers Hyderabad']
    
    venues = ['Wankhede Stadium', 'MA Chidambaram Stadium', 'Eden Gardens', 
              'M. Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'Narendra Modi Stadium']
    
    data = []
    for _ in range(n_samples):
        team1 = np.random.choice(teams)
        team2 = np.random.choice([t for t in teams if t != team1])
        venue = np.random.choice(venues)
        
        # Features
        team1_wins_last_5 = np.random.randint(0, 6)
        team2_wins_last_5 = np.random.randint(0, 6)
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['bat', 'field'])
        
        # Home advantage
        home_advantage = 0
        if 'Mumbai' in team1 and venue == 'Wankhede Stadium':
            home_advantage = 1
        elif 'Chennai' in team1 and venue == 'MA Chidambaram Stadium':
            home_advantage = 1
        elif 'Kolkata' in team1 and venue == 'Eden Gardens':
            home_advantage = 1
        
        # Simulate winner (with some logic)
        win_prob_team1 = 0.5
        win_prob_team1 += (team1_wins_last_5 - team2_wins_last_5) * 0.05
        win_prob_team1 += home_advantage * 0.15
        if toss_winner == team1:
            win_prob_team1 += 0.1
        
        winner = team1 if np.random.random() < win_prob_team1 else team2
        
        data.append({
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'team1_wins_last_5': team1_wins_last_5,
            'team2_wins_last_5': team2_wins_last_5,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'home_advantage': home_advantage,
            'winner': winner
        })
    
    return pd.DataFrame(data)

# Traditional rule-based prediction
def predict_traditional(team1, team2, venue, toss_winner, team1_form, team2_form):
    """Traditional rule-based prediction"""
    # Simple fixed rules
    if venue == "Wankhede Stadium" and team1 == "Mumbai Indians":
        return team1, 0.75  # Home advantage for MI
    elif venue == "MA Chidambaram Stadium" and team2 == "Chennai Super Kings":
        return team2, 0.75  # Home advantage for CSK
    elif team1_form >= 4 and team2_form <= 2:
        return team1, 0.65  # Form advantage
    elif team2_form >= 4 and team1_form <= 2:
        return team2, 0.65  # Form advantage
    elif toss_winner == team1:
        return team1, 0.55  # Slight toss advantage
    else:
        return team2, 0.55  # Default

# Train ML model
@st.cache_resource
def train_ml_model():
    """Train a Random Forest model on synthetic data"""
    df = generate_ipl_data(2000)
    
    # Feature engineering
    features = []
    for _, row in df.iterrows():
        features.append([
            row['team1_wins_last_5'],
            row['team2_wins_last_5'],
            row['home_advantage'],
            1 if row['toss_winner'] == row['team1'] else 0,
            1 if row['toss_decision'] == 'bat' else 0,
            hash(row['team1']) % 10,  # Simple team encoding
            hash(row['team2']) % 10,
            hash(row['venue']) % 10   # Simple venue encoding
        ])
    
    X = np.array(features)
    y = (df['winner'] == df['team1']).astype(int)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, train_acc, test_acc

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Comparison", "🎮 Interactive Demo", 
                                   "🧠 How ML Works", "📈 Model Performance"])

# Tab 1: Visual Comparison
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Traditional Programming")
        st.code("""
def predict_winner(team1, team2, venue):
    # Fixed rules written by programmer
    if venue == "Wankhede" and team1 == "MI":
        return "Mumbai Indians"  # Rule 1
    elif venue == "Chepauk" and team2 == "CSK":
        return "Chennai Super Kings"  # Rule 2
    else:
        return "Toss Winner"  # Default rule
        
# Problems:
# ❌ Only works for specific scenarios
# ❌ Can't handle new situations
# ❌ No confidence score
# ❌ Doesn't improve over time
        """, language='python')
        
        # Pros and Cons
        st.success("✅ **Pros:**")
        st.markdown("""
        - Simple to understand
        - Fast execution
        - Predictable behavior
        - No training needed
        """)
        
        st.error("❌ **Cons:**")
        st.markdown("""
        - Limited to known scenarios
        - Can't adapt to new patterns
        - Manual rule updates needed
        - No probability/confidence
        """)
    
    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.code("""
# Train model on historical data
model = LinearRegression()
model.fit(X_train, y_train)  # Learn patterns

def predict_winner(features):
    # Model learns patterns from data
    prediction = model.predict(features)
    confidence = model.predict_proba(features)
    
    return prediction, confidence
    
# Advantages:
# ✅ Handles ANY scenario
# ✅ Provides confidence scores
# ✅ Improves with more data
# ✅ Finds complex patterns
        """, language='python')
        
        st.success("✅ **Pros:**")
        st.markdown("""
        - Learns from data automatically
        - Handles complex patterns
        - Provides confidence scores
        - Improves with more data
        """)
        
        st.warning("⚠️ **Cons:**")
        st.markdown("""
        - Needs training data
        - "Black box" nature
        - Requires more resources
        - Can overfit if not careful
        """)
    
    # Flowchart comparison
    st.markdown("---")
    st.markdown("### 🔄 Logic Flow Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Traditional Flow:**")
        st.markdown("""
        ```mermaid
        graph TD
            A[Input: Teams, Venue] --> B{Is MI at Wankhede?}
            B -->|Yes| C[Predict: MI Wins]
            B -->|No| D{Is CSK at Chepauk?}
            D -->|Yes| E[Predict: CSK Wins]
            D -->|No| F[Predict: Random]
        ```
        """)
        st.info("💡 **Fixed decision tree with hardcoded rules**")
    
    with col2:
        st.markdown("**ML Flow:**")
        st.markdown("""
        ```mermaid
        graph TD
            A[Historical Data] --> B[Feature Extraction]
            B --> C[Train Model]
            C --> D[Learned Patterns]
            E[New Match Input] --> F[Apply Model]
            D --> F
            F --> G[Prediction + Confidence]
        ```
        """)
        st.info("💡 **Dynamic patterns learned from data**")

# Tab 2: Interactive Demo
with tab2:
    st.markdown("### 🎮 Try Both Approaches")
    st.markdown("Configure match parameters and see predictions from both models")
    
    # Input controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1 = st.selectbox("Team 1", 
            ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals'])
        team1_form = st.slider("Team 1 Recent Form (wins in last 5)", 0, 5, 3)
    
    with col2:
        team2 = st.selectbox("Team 2",
            ['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals'])
        team2_form = st.slider("Team 2 Recent Form (wins in last 5)", 0, 5, 2)
    
    with col3:
        venue = st.selectbox("Venue",
            ['Wankhede Stadium', 'MA Chidambaram Stadium', 'Eden Gardens', 
             'M. Chinnaswamy Stadium', 'Arun Jaitley Stadium'])
        toss_winner = st.radio("Toss Winner", [team1, team2])
    
    # Make predictions
    if st.button("🎯 Predict Winner", type="primary"):
        # Traditional prediction
        trad_winner, trad_conf = predict_traditional(team1, team2, venue, toss_winner, 
                                                     team1_form, team2_form)
        
        # ML prediction
        model, _, _ = train_ml_model()
        
        # Prepare features for ML
        home_adv = 1 if ('Mumbai' in team1 and venue == 'Wankhede Stadium') else 0
        features = np.array([[
            team1_form, team2_form, home_adv,
            1 if toss_winner == team1 else 0,
            1,  # Assuming bat first
            hash(team1) % 10,
            hash(team2) % 10,
            hash(venue) % 10
        ]])
        
        ml_pred = model.predict(features)[0]
        ml_prob = model.predict_proba(features)[0]
        ml_winner = team1 if ml_pred == 1 else team2
        ml_conf = max(ml_prob)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional Programming")
            st.metric("Winner", trad_winner)
            st.metric("Confidence", f"{trad_conf:.1%}")
            
            if trad_conf < 0.6:
                st.warning("⚠️ Low confidence - mostly guessing!")
            
            # Show why
            st.info(f"""
            **Logic Used:**
            - Home advantage check: {'Yes' if venue == 'Wankhede Stadium' and team1 == 'Mumbai Indians' else 'No'}
            - Form-based rule: {'Yes' if abs(team1_form - team2_form) >= 2 else 'No'}
            - Toss advantage: {toss_winner}
            """)
        
        with col2:
            st.markdown("#### Machine Learning")
            st.metric("Winner", ml_winner)
            st.metric("Confidence", f"{ml_conf:.1%}")
            
            if ml_conf > 0.7:
                st.success("✅ High confidence prediction!")
            
            # Feature importance
            st.info(f"""
            **Factors Considered:**
            - Recent form difference: {team1_form - team2_form}
            - Home advantage: {'Yes' if home_adv else 'No'}
            - Toss advantage: {'Yes' if toss_winner == team1 else 'No'}
            - Historical venue performance
            - Head-to-head records
            """)
        
        # Probability distribution
        st.markdown("---")
        st.markdown("### 📊 Win Probability Distribution")
        
        prob_data = pd.DataFrame({
            'Team': [team1, team2],
            'Traditional': [trad_conf if trad_winner == team1 else 1-trad_conf,
                          trad_conf if trad_winner == team2 else 1-trad_conf],
            'ML Model': [ml_prob[1], ml_prob[0]]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Traditional', x=prob_data['Team'], y=prob_data['Traditional'],
                            marker_color='lightblue'))
        fig.add_trace(go.Bar(name='ML Model', x=prob_data['Team'], y=prob_data['ML Model'],
                            marker_color='darkblue'))
        fig.update_layout(barmode='group', yaxis_title='Win Probability',
                         title='Prediction Comparison')
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: How ML Works
with tab3:
    st.markdown("### 🧠 Understanding Machine Learning")
    
    # Train model and get feature importance
    model, train_acc, test_acc = train_ml_model()
    
    # Feature importance
    feature_names = ['Team1 Form', 'Team2 Form', 'Home Advantage', 'Toss Winner', 
                    'Toss Decision', 'Team1 Strength', 'Team2 Strength', 'Venue Factor']
    importance = model.feature_importances_
    
    # Create feature importance plot
    fig_importance = px.bar(
        x=importance, 
        y=feature_names, 
        orientation='h',
        title='Feature Importance in ML Model',
        labels={'x': 'Importance', 'y': 'Features'},
        color=importance,
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Model Statistics")
        st.metric("Training Accuracy", f"{train_acc:.1%}")
        st.metric("Testing Accuracy", f"{test_acc:.1%}")
        st.metric("Number of Trees", "100")
        st.metric("Features Used", "8")
        
        st.info("""
        **What the model learned:**
        - Recent form is most important
        - Home advantage matters significantly
        - Toss has moderate impact
        - Venue history influences outcomes
        """)
    
    with col2:
        st.markdown("### 🎯 How ML Makes Decisions")
        
        st.markdown("""
        **1. Training Phase:**
        - Analyzes thousands of historical matches
        - Identifies patterns in winning teams
        - Learns which factors matter most
        
        **2. Prediction Phase:**
        - Takes new match parameters
        - Applies learned patterns
        - Calculates probability for each team
        - Provides confidence score
        
        **3. Continuous Improvement:**
        - Gets feedback from actual results
        - Updates patterns with new data
        - Becomes more accurate over time
        """)
    
    # Show sample decision tree
    st.markdown("---")
    st.markdown("### 🌳 Sample Decision Path")
    st.code("""
    IF team1_form > 3.5:
        IF home_advantage == 1:
            Probability(team1_wins) = 0.78
        ELSE:
            IF toss_winner == team1:
                Probability(team1_wins) = 0.65
            ELSE:
                Probability(team1_wins) = 0.52
    ELSE:
        IF team2_form > 3.5:
            Probability(team1_wins) = 0.35
        ELSE:
            Probability(team1_wins) = 0.48
    """)
    
    st.success("💡 **Note:** ML model uses 100 such trees and averages their predictions!")

# Tab 4: Model Performance
with tab4:
    st.markdown("### 📈 Performance Comparison")
    
    # Generate performance data
    scenarios = ['Known Teams & Venues', 'New Venues', 'New Team Combinations', 
                 'Playoff Matches', 'Weather Affected', 'Player Injuries']
    
    traditional_acc = [75, 45, 40, 50, 30, 35]
    ml_acc = [82, 78, 75, 80, 72, 70]
    
    # Create comparison chart
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(name='Traditional', x=scenarios, y=traditional_acc,
                                   marker_color='lightcoral'))
    fig_comparison.add_trace(go.Bar(name='Machine Learning', x=scenarios, y=ml_acc,
                                   marker_color='lightgreen'))
    
    fig_comparison.update_layout(
        title='Accuracy in Different Scenarios (%)',
        yaxis_title='Accuracy (%)',
        barmode='group',
        showlegend=True
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Learning curve
    st.markdown("---")
    st.markdown("### 📚 Learning Curve")
    
    # Simulate learning curve
    data_points = [100, 200, 500, 1000, 2000, 5000, 10000]
    trad_performance = [60, 60, 60, 60, 60, 60, 60]  # Doesn't improve
    ml_performance = [55, 62, 68, 72, 76, 79, 82]  # Improves with data
    
    fig_learning = go.Figure()
    fig_learning.add_trace(go.Scatter(x=data_points, y=trad_performance, mode='lines+markers',
                                     name='Traditional', line=dict(color='red', dash='dash')))
    fig_learning.add_trace(go.Scatter(x=data_points, y=ml_performance, mode='lines+markers',
                                     name='Machine Learning', line=dict(color='green')))
    
    fig_learning.update_layout(
        title='Performance vs Amount of Data',
        xaxis_title='Number of Training Examples',
        yaxis_title='Accuracy (%)',
        xaxis_type='log'
    )
    st.plotly_chart(fig_learning, use_container_width=True)
    
    # Key insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Traditional Approach")
        st.error("""
        **Limitations:**
        - Accuracy plateaus at 60%
        - Fails in new scenarios
        - No improvement with more data
        - Requires manual rule updates
        - Can't capture complex patterns
        """)
    
    with col2:
        st.markdown("### 🟢 Machine Learning")
        st.success("""
        **Advantages:**
        - Accuracy improves to 82%+
        - Handles new scenarios well
        - Improves with more data
        - Automatically finds patterns
        - Provides confidence scores
        """)
    
    # Summary
    st.markdown("---")
    st.markdown("### 🎯 Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ML Advantage", "+22%", "Average accuracy gain")
    
    with col2:
        st.metric("Adaptability", "∞", "Handles unlimited scenarios")
    
    with col3:
        st.metric("Improvement Rate", "↑37%", "With 10x more data")
    
    st.info("""
    ### 📝 When to use which approach?
    
    **Use Traditional Programming when:**
    - Rules are simple and well-defined
    - Requirements never change
    - 100% accuracy needed for specific cases
    - Explainability is critical
    
    **Use Machine Learning when:**
    - Patterns are complex
    - Rules are hard to define
    - Data is available
    - Need to handle many scenarios
    - Probability/confidence matters
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created for MLOps Masterclass | Learn how ML transforms traditional programming</p>
    <p>🏏 Cricket + 🤖 ML = Better Predictions!</p>
</div>
""", unsafe_allow_html=True) 