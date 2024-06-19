import streamlit as st
from ui import set_page_container_style, sidebar
import sys
import os

# Add the parent directory to the sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprocessing.dataprocessing import load_and_process_data
from explo import stat_explo
from model import modelisation

# Set the style for the container
set_page_container_style()

# Sidebar navigation
page = sidebar()

# Load data
data = load_and_process_data('../data/processed/combined_nba_stats_2021_2024.csv')

# Filter data to include only players with more than 15 minutes per game
filtered_data = stat_explo.filter_data_by_play_time(data, 15)

# Display page based on selection
if page == "Stats exploratoires":
    st.title("NBA Player Stats Analysis")
    stat_explo.show_tabs(filtered_data)
elif page == "Modélisation":
    st.title("Modélisation")
    modelisation.main()
