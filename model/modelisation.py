import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
from model.bmodel import load_and_prepare_data, train_svm_model, svm

# Define NBA colors
NBA_BLUE = "#1D428A"
NBA_RED = "#C8102E"

# Load and prepare data
filepath = "../data/processed/combined_nba_stats_2021_2024.csv"
X, y, column_names = load_and_prepare_data(filepath)
model, X_test, y_test = train_svm_model(X, y)

# Define labels for the columns
labels = {
    'Age': 'Player\'s age',
    'G': 'Games played',
    'GS': 'Games started',
    'MP': 'Minutes played per game',
    'FG': 'Field goals per game',
    'FGA': 'Field goal attempts per game',
    'FG%': 'Field goal percentage',
    '3P': '3-point field goals per game',
    '3PA': '3-point field goal attempts per game',
    '3P%': '3-point field goal percentage',
    '2P': '2-point field goals per game',
    '2PA': '2-point field goal attempts per game',
    '2P%': '2-point field goal percentage',
    'eFG%': 'Effective field goal percentage',
    'FT': 'Free throws per game',
    'FTA': 'Free throw attempts per game',
    'FT%': 'Free throw percentage',
    'ORB': 'Offensive rebounds per game',
    'DRB': 'Defensive rebounds per game',
    'TRB': 'Total rebounds per game',
    'AST': 'Assists per game',
    'STL': 'Steals per game',
    'BLK': 'Blocks per game',
    'TOV': 'Turnovers per game',
    'PF': 'Personal fouls per game',
    'PTS': 'Points per game'
}

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=class_names, y=class_names)
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        coloraxis_showscale=False,
        font=dict(color=NBA_BLUE),
        xaxis=dict(tickmode='array', tickvals=list(range(len(class_names))), ticktext=class_names)
    )
    st.plotly_chart(fig)

def plot_prediction_vs_actual(y_true, y_pred):
    df_results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    df_results['Correct'] = df_results['Actual'] == df_results['Predicted']

    correct_counts = df_results.groupby(['Actual', 'Correct']).size().reset_index(name='Count')
    fig = px.bar(correct_counts, x='Actual', y='Count', color='Correct', barmode='group',
                 color_discrete_map={True: NBA_BLUE, False: NBA_RED})
    fig.update_layout(title="Correct vs Incorrect Predictions by Position", font=dict(color=NBA_BLUE), showlegend=False)
    st.plotly_chart(fig)

def manual_prediction_input(column_names, labels):
    st.header("Manual Input for Prediction")

    input_data = {}
    computed_data = {}

    for col in column_names:
        if col not in ['FG%', '3P%', '2P%', 'eFG%', 'FT%']:  # Skip computed fields
            label = labels.get(col, col)
            input_data[col] = st.number_input(f"Enter value for {label}", min_value=float(X[col].min()), max_value=float(X[col].max()))

    # Compute the percentage fields
    if 'FG' in input_data and 'FGA' in input_data and input_data['FGA'] != 0:
        computed_data['FG%'] = input_data['FG'] / input_data['FGA']
    if '3P' in input_data and '3PA' in input_data and input_data['3PA'] != 0:
        computed_data['3P%'] = input_data['3P'] / input_data['3PA']
    if '2P' in input_data and '2PA' in input_data and input_data['2PA'] != 0:
        computed_data['2P%'] = input_data['2P'] / input_data['2PA']
    if 'FG' in input_data and 'FGA' in input_data and input_data['FGA'] != 0:
        computed_data['eFG%'] = (input_data['FG'] + 0.5 * input_data['3P']) / input_data['FGA'] if '3P' in input_data else input_data['FG'] / input_data['FGA']
    if 'FT' in input_data and 'FTA' in input_data and input_data['FTA'] != 0:
        computed_data['FT%'] = input_data['FT'] / input_data['FTA']

    for col, value in computed_data.items():
        st.write(f"{labels.get(col, col)}: {value:.2f}")

    input_data.update(computed_data)
    input_df = pd.DataFrame([input_data])
    
    # Ensure all necessary columns are present and in the correct order
    for col in column_names:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[column_names]
    
    prediction = svm(input_df, model)
    st.write(f"Predicted Position: {prediction[0]}")

def main():
    st.title("NBA Player Position Prediction")

    # Display performance graphs
    y_pred = model.predict(X_test)
    class_names = model.classes_

    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred, class_names)

    st.subheader("Correct vs Incorrect Predictions by Position")
    plot_prediction_vs_actual(y_test, y_pred)

    # Manual prediction input
    manual_prediction_input(column_names, labels)
