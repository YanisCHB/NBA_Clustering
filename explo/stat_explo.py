import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, delimiter=';')

def plot_histogram(data: pd.DataFrame, column: str):
    fig = px.histogram(data, x=column)
    st.plotly_chart(fig)

def plot_scatter(data: pd.DataFrame, x: str, y: str, color: str = None):
    fig = px.scatter(data, x=x, y=y, color=color)
    st.plotly_chart(fig)

def plot_bar(data: pd.DataFrame, x: str, y: str):
    fig = px.bar(data, x=x, y=y)
    st.plotly_chart(fig)

def plot_heatmap(data: pd.DataFrame):
    # Select only the specified columns for the heatmap
    stats_columns = [
        'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
        'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
    ]
    
    # Calculate the average values of each statistic for each position
    avg_stats_by_pos = data.groupby('Pos')[stats_columns].mean()
    
    # Generate a heatmap figure
    fig = ff.create_annotated_heatmap(
        z=avg_stats_by_pos.values,
        x=stats_columns,
        y=avg_stats_by_pos.index.tolist(),
        annotation_text=avg_stats_by_pos.round(2).values,
        showscale=True
    )
    
    # Display the heatmap
    st.plotly_chart(fig)

def show_exploratory_stats(data: pd.DataFrame):
    st.title("Exploratory Statistics")

    st.header("Histogram of Points")
    plot_histogram(data, 'PTS')

    st.header("Scatter plot of Points vs Assists")
    plot_scatter(data, x='PTS', y='AST', color='Pos')

    st.header("Bar plot of Average Points by Position")
    avg_points_by_pos = data.groupby('Pos')['PTS'].mean().reset_index()
    plot_bar(avg_points_by_pos, x='Pos', y='PTS')

    st.header("Heatmap of Average Stats by Position")
    plot_heatmap(data)
