import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

NBA_BLUE = "#1D428A"
NBA_RED = "#C8102E"

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, delimiter=';')

@st.cache_data
def filter_data_by_play_time(data: pd.DataFrame, min_play_time: float) -> pd.DataFrame:
    return data[data['MP'] >= min_play_time]

def plot_histogram(data: pd.DataFrame, column: str):
    fig = px.histogram(data, x=column, color_discrete_sequence=[NBA_BLUE])
    st.plotly_chart(fig)

def plot_scatter(data: pd.DataFrame, x: str, y: str, color: str = None):
    fig = px.scatter(data, x=x, y=y, color=color, color_discrete_sequence=[NBA_BLUE])
    st.plotly_chart(fig)

def plot_bar(data: pd.DataFrame, x: str, y: str):
    fig = px.bar(data, x=x, y=y, color_discrete_sequence=[NBA_BLUE])
    st.plotly_chart(fig)

def plot_heatmap(data: pd.DataFrame):
    data = data.drop(columns=['Rk'])  # Remove the Rank column
    numeric_data = data.select_dtypes(include=[float, int])  # Select only numeric columns
    corr_matrix = numeric_data.corr()
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Reds')
    fig.update_layout(
        title="Correlation Heatmap",
        font=dict(color=NBA_BLUE),
        width=900,  # Adjust the width as needed
        height=800,  # Adjust the height as needed
        margin=dict(l=100, r=100, t=100, b=100),  # Center the heatmap
        xaxis={'side': 'top'}  # Move x-axis to the top
    )
    st.plotly_chart(fig, use_container_width=True)

def show_exploratory_stats(data: pd.DataFrame):
    st.title("Exploratory Statistics")
    st.header("Histogram of Points")
    plot_histogram(data, 'PTS')

    st.header("Scatter plot of Points vs Assists")
    plot_scatter(data, x='PTS', y='AST', color='Pos')

    st.header("Bar plot of Average Points by Position")
    avg_points_by_pos = data.groupby('Pos')['PTS'].mean().reset_index()
    plot_bar(avg_points_by_pos, x='Pos', y='PTS')

    st.header("Heatmap of Correlation Stats")
    plot_heatmap(data)

def show_position_stats(data: pd.DataFrame, pos: str):
    stats_columns = [
        'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
        'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
    ]
    labels = {
        'FG': 'Field goals per game', 'FGA': 'Field goal attempts per game', 'FG%': 'Field goal percentage',
        '3P': '3-point field goals per game', '3PA': '3-point field goal attempts per game', '3P%': '3-point field goal percentage',
        '2P': '2-point field goals per game', '2PA': '2-point field goal attempts per game', '2P%': '2-point field goal percentage',
        'eFG%': 'Effective field goal percentage', 'FT': 'Free throws per game', 'FTA': 'Free throw attempts per game',
        'FT%': 'Free throw percentage', 'ORB': 'Offensive rebounds per game', 'DRB': 'Defensive rebounds per game',
        'TRB': 'Total rebounds per game', 'AST': 'Assists per game', 'STL': 'Steals per game', 'BLK': 'Blocks per game',
        'TOV': 'Turnovers per game', 'PF': 'Personal fouls per game', 'PTS': 'Points per game'
    }
    filtered_data = data[data['Pos'] == pos]
    avg_stats = filtered_data[stats_columns].mean().reset_index()
    avg_stats.columns = ['Statistic', 'Average']
    avg_stats['Statistic'] = avg_stats['Statistic'].map(labels)
    st.table(avg_stats.set_index('Statistic'))

def show_stats_ranking(data: pd.DataFrame, stat: str):
    labels = {
        'G': 'Games played', 'GS': 'Games started', 'MP': 'Minutes played per game',
        'FG': 'Field goals per game', 'FGA': 'Field goal attempts per game', 'FG%': 'Field goal percentage',
        '3P': '3-point field goals per game', '3PA': '3-point field goal attempts per game', '3P%': '3-point field goal percentage',
        '2P': '2-point field goals per game', '2PA': '2-point field goal attempts per game', '2P%': '2-point field goal percentage',
        'eFG%': 'Effective field goal percentage', 'FT': 'Free throws per game', 'FTA': 'Free throw attempts per game',
        'FT%': 'Free throw percentage', 'ORB': 'Offensive rebounds per game', 'DRB': 'Defensive rebounds per game',
        'TRB': 'Total rebounds per game', 'AST': 'Assists per game', 'STL': 'Steals per game', 'BLK': 'Blocks per game',
        'TOV': 'Turnovers per game', 'PF': 'Personal fouls per game', 'PTS': 'Points per game'
    }
    filtered_data = data.groupby('Pos')[stat].mean().reset_index()
    filtered_data = filtered_data.sort_values(by=stat, ascending=False)
    filtered_data.columns = ['Position', 'Average']
    filtered_data['Statistic'] = labels[stat]
    fig = px.bar(filtered_data, x='Average', y='Position', orientation='h', color='Position', color_discrete_sequence=[NBA_BLUE])
    fig.update_layout(title=f"Average {labels[stat]} by Position", font=dict(color=NBA_BLUE), showlegend=False)
    st.plotly_chart(fig)

def show_tabs(data: pd.DataFrame):
    tab1, tab2 = st.tabs(["Postes", "Stats"])

    with tab1:
        position_labels = {
            'SG': 'Shooting Guard', 'SF': 'Small Forward', 'PG': 'Point Guard', 'PF': 'Power Forward', 'C': 'Center',
            'SG-SF': 'Shooting Guard/Small Forward', 'SF-SG': 'Small Forward/Shooting Guard',
            'SG-PG': 'Shooting Guard/Point Guard', 'PG-SG': 'Point Guard/Shooting Guard',
            'PF-SF': 'Power Forward/Small Forward', 'SF-PF': 'Small Forward/Power Forward',
            'C-PF': 'Center/Power Forward', 'PF-C': 'Power Forward/Center'
        }
        positions = data['Pos'].unique()
        position_options = {position: position_labels.get(position, position) for position in positions}
        pos = st.selectbox("Select Position", list(position_options.values()), format_func=lambda x: x)
        pos_key = list(position_options.keys())[list(position_options.values()).index(pos)]
        show_position_stats(data, pos_key)

    with tab2:
        stat_labels = {
            'G': 'Games played', 'GS': 'Games started', 'MP': 'Minutes played per game',
            'FG': 'Field goals per game', 'FGA': 'Field goal attempts per game', 'FG%': 'Field goal percentage',
            '3P': '3-point field goals per game', '3PA': '3-point field goal attempts per game', '3P%': '3-point field goal percentage',
            '2P': '2-point field goals per game', '2PA': '2-point field goal attempts per game', '2P%': '2-point field goal percentage',
            'eFG%': 'Effective field goal percentage', 'FT': 'Free throws per game', 'FTA': 'Free throw attempts per game',
            'FT%': 'Free throw percentage', 'ORB': 'Offensive rebounds per game', 'DRB': 'Defensive rebounds per game',
            'TRB': 'Total rebounds per game', 'AST': 'Assists per game', 'STL': 'Steals per game', 'BLK': 'Blocks per game',
            'TOV': 'Turnovers per game', 'PF': 'Personal fouls per game', 'PTS': 'Points per game'
        }
        stat = st.selectbox("Select Stat", list(stat_labels.values()), format_func=lambda x: x)
        stat_key = list(stat_labels.keys())[list(stat_labels.values()).index(stat)]
        show_stats_ranking(data, stat_key)
        st.header("Heatmap of Correlation Stats")
        plot_heatmap(data)
