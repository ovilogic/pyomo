import pandas as pd

from pathlib import Path

p = Path(__file__)

DATA = p.resolve().parents[1] / "data" / "OrangeJuiceBlending.xlsx"

print(f"Loading data from {DATA}")

df = pd.read_excel(DATA, sheet_name="Optimization Model (Limit 4)", header=None)

demand_row = df.index[df.iloc[:, 0] == "Total Required"][0]
demand = {
    "January": df.iloc[demand_row, 2],
    "February": df.iloc[demand_row, 3],
    "March": df.iloc[demand_row, 4],
}

Valencia_row = df.index[df.iloc[:, 0] == "Valencia Required"][0]
Valencia_req = {
    "January": df.iloc[Valencia_row, 2],
    "February": df.iloc[Valencia_row, 3],
    "March": df.iloc[Valencia_row, 4],
}

print(df.iloc[25:35, 0:5])

# Quality constraints
qc_row = df.index[df.iloc[:, 0] == "Quality Constraints"][0]
BAR_min = df.iloc[qc_row + 1, 1]
BAR_max = df.iloc[qc_row + 1, 5]
ACID_min = df.iloc[qc_row + 2, 1]
ACID_max = df.iloc[qc_row + 2, 5]   
ASTRINGENCY_min = df.iloc[qc_row + 3, 1]
ASTRINGENCY_max = df.iloc[qc_row + 3, 5]
COLOR_min = df.iloc[qc_row + 4, 1]
COLOR_max = df.iloc[qc_row + 4, 5]

quality = {
    "BAR": (BAR_min, BAR_max),
    "ACID": (ACID_min, ACID_max),
    "ASTRINGENCY": (ASTRINGENCY_min, ASTRINGENCY_max),
    "COLOR": (COLOR_min, COLOR_max),
}   

