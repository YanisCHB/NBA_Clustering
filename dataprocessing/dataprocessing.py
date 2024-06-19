import pandas as pd

# Lire et concaténer les fichiers CSV en ajoutant directement la colonne Year
def read_and_concat_csv_files(files_and_years):
    all_dfs = []
    for file_path, year in files_and_years:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', sep=';')
        df['Year'] = year
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

# Nettoyer les données des positions
def clean_positions(data):
    data['Pos'] = data['Pos'].replace({
        'SG-SF': 'SG-SF', 'SF-SG': 'SG-SF',
        'SG-PG': 'SG-PG', 'PG-SG': 'SG-PG',
        'PF-SF': 'PF-SF', 'SF-PF': 'PF-SF',
        'C-PF': 'C-PF', 'PF-C': 'C-PF'
    })
    return data

def load_and_process_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, delimiter=';')
    return clean_positions(data)

# Liste des fichiers et des années
files_and_years = [
    ('../data/raw/2021-2022 NBA Player Stats.csv', '2021-2022'),
    ('../data/raw/2022-2023 NBA Player Stats.csv', '2022-2023'),
    ('../data/raw/2023-2024 NBA Player Stats.csv', '2023-2024')
]

# Lire, concaténer et nettoyer les fichiers CSV
try:
    combined_df = read_and_concat_csv_files(files_and_years)
    
    # Nettoyer les positions
    cleaned_df = clean_positions(combined_df)
    
    # Enregistrer le DataFrame combiné et nettoyé dans un nouveau fichier CSV dans le répertoire ../data/processed/
    output_file_path = '../data/processed/combined_nba_stats_2021_2024.csv'
    cleaned_df.to_csv(output_file_path, index=False, sep=';')
    print(f"Les données des saisons 2021-2022, 2022-2023 et 2023-2024 ont été combinées et nettoyées dans {output_file_path}")
except FileNotFoundError as e:
    print(f"Erreur: {e}")
except Exception as e:
    print(f"Erreur inattendue: {e}")
