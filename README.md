# NBA Player Stats Analysis and Prediction

This project involves data mining and machine learning on NBA player statistics. It includes exploratory data analysis and a classification model to predict player positions based on their stats.

## Project Structure

├── app
│ ├── app.py
│ ├── pycache
│ └── ui.py
├── data
│ ├── processed
│ │ └── combined_nba_stats_2021_2024.csv
│ └── raw
│ ├── 2021-2022 NBA Player Stats.csv
│ ├── 2022-2023 NBA Player Stats.csv
│ └── 2023-2024 NBA Player Stats.csv
├── dataprocessing
│ ├── dataprocessing.py
│ └── processing.py
├── explo
│ └── stat_explo.py
├── model
│ ├── bmodel.py
│ └── modelisation.py
├── nba.png
└── README.md


## Setup

### Prerequisites

- Python 3.8 or later
- pip (Python package installer)
- Virtual environment (recommended)

### Installation


1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/nba_stats_analysis.git
   cd nba_stats_analysis
   ```


2. **Create and activate virtual environment:**

```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install required packages:**

```sh
pip install -r requirements.txt
```

## Data Preparation

Ensure that the raw data files are placed in the data/raw/ directory:

    2021-2022 NBA Player Stats.csv
    2022-2023 NBA Player Stats.csv
    2023-2024 NBA Player Stats.csv

## Running the Project

1. **Preprocess the data:**

    The dataprocessing.py script reads and processes the raw data files, merging them into a single CSV file in the data/processed/ directory.


```sh
python dataprocessing/dataprocessing.py
```

This will generate the combined_nba_stats_2021_2024.csv file in the data/processed/ directory.

2. **Run the Streamlit application:**

Use the streamlit command to start the app:

```sh
streamlit run app/app.py
```

3. **Access the application:**

Open your web browser and go to the URL displayed in the terminal, typically http://localhost:8501.

