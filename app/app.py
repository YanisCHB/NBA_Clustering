import streamlit as st
from ui import set_page_container_style, sidebar
import sys
import os

# Add the parent directory to the sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explo import stat_explo
from dataprocessing.dataprocessing import load_and_process_data

# Set the style for the container
set_page_container_style()

# Sidebar navigation
page = sidebar()

# Load data
data = load_and_process_data('../data/processed/combined_nba_stats_2021_2024.csv')

# Display page based on selection
if page == "Stats exploratoires":
    stat_explo.show_exploratory_stats(data)
elif page == "Modélisation":
    st.title("Modélisation")
    st.write("La partie modélisation sera ajoutée plus tard.")
