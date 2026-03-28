import pandas as pd

from pathlib import Path
from pyomo.environ import *

p = Path(__file__)

DATA = p.resolve().parents[1] / "data" / "OrangeJuiceBlending.xlsx"

df = pd.read_excel(DATA, sheet_name="Optimization Model (Limit 4)", header=None)


def load_supplier_data(df):
    # Find the row where the supplier data starts
    supplier_headers = df.index[df.iloc[:, 0] == "Varietal"][0]
    supplier_bottom = df.index[df.iloc[:, 0] == "Monthly Cost Totals:"][0]
    suppliers_df = df.iloc[supplier_headers + 1 : supplier_bottom, 0:13]
    # suppliers_cols = df.iloc[supplier_headers, 0:13].tolist()
    
    return suppliers_df

def load_demand_and_quality_constraints(df):
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
    return demand, Valencia_req, quality

def build_basic_model(suppliers_df, demand, Valencia_req, quality):
    model = ConcreteModel()

    # Define sets
    model.SUPPLIERS = Set(initialize=suppliers_df.index.tolist())
    model.MONTHS = Set(initialize=["January", "February", "March"])

    # Decision variables
    model.x = Var(model.SUPPLIERS, model.MONTHS, domain=NonNegativeReals)

    # Parameters
    available = suppliers_df.iloc[:, 6].tolist()
    print(available)
    brix_acid = suppliers_df.iloc[:, 7].tolist()
    acid = suppliers_df.iloc[:, 8].tolist()
    astringency = suppliers_df.iloc[:, 9].tolist()
    color = suppliers_df.iloc[:, 10].tolist()
    price = suppliers_df.iloc[:, 11].tolist()
    shipping = suppliers_df.iloc[:, 12].tolist()
    
    # Objective function
    cost = 0
    for s in model.SUPPLIERS:
        for m in model.MONTHS:
            idx = s - list(model.SUPPLIERS)[0]
            cost += model.x[s, m] * (price[idx] + shipping[idx])
    model.OBJ = Objective(expr=cost, sense=minimize)

    # Constraints
    ## Supply constraints
    ### Each month must meet a specific demand
    # demand_jan = 0
    # demand_feb = 0
    # demand_mar = 0
    # for s in model.SUPPLIERS:
    #     for m in model.MONTHS:
    #         if m == "January":
    #             demand_jan += model.x[s, m]
    #         elif m == "February":
    #             demand_feb += model.x[s, m]
    #         else:
    #             demand_mar += model.x[s, m]
    # model.Demand_Jan = Constraint(expr=demand_jan >= demand["January"])
    # model.Demand_Feb = Constraint(expr=demand_feb >= demand["February"])
    # model.Demand_Mar = Constraint(expr=demand_mar >= demand["March"])

    # Alternative way to write demand constraints with summation and generators
    model.Demand_Jan_Alt = Constraint(expr=sum(model.x[s, "January"] for s in model.SUPPLIERS) >= demand["January"])
    model.Demand_Feb_Alt = Constraint(expr=sum(model.x[s, "February"] for s in model.SUPPLIERS) >= demand["February"])
    model.Demand_Mar_Alt = Constraint(expr=sum(model.x[s, "March"] for s in model.SUPPLIERS) >= demand["March"])

    # Constraints written with rules and functions
    ## Each month must meet a specific demand
    def demand_function(model, m):
        return sum(model.x[s, m] for s in model.SUPPLIERS) == demand[m]
    model.demand_constraint = Constraint(model.MONTHS, rule=demand_function)

    ## Valencia content constraints. Valencia orders for each month must make up at least 40% of the total blend.
    def valencia_constraint(model, m):
        # Supplier at row 7 is Valencia
        return model.x[7, m] >= Valencia_req[m]
    model.Valencia_constraint = Constraint(model.MONTHS, rule=valencia_constraint)

    ## Quality constraints
    jan_brix_acid = 0
    for s in model.SUPPLIERS:
        for m in model.MONTHS:
            idx = s - list(model.SUPPLIERS)[0]
            jan_brix_acid += model.x[s, m] * suppliers_df.iloc[idx, 7] 
    jan_brix_acid_ratio = jan_brix_acid / demand["January"]
    model.Jan_Brix_Min = Constraint(expr=jan_brix_acid_ratio >= quality["BAR"][0])
    model.Jan_Brix_Max = Constraint(expr=jan_brix_acid_ratio <= quality["BAR"][1])

    # Suboptimal way to define constraints for all quality parameters and months using loops and if statements
    # def brix(model, m):
    #     jan_brix_ratio = sum(model.x[s, m] * suppliers_df.iloc[s - list(model.SUPPLIERS)[0], 7] for s in model.SUPPLIERS) / demand[m]
    #     return quality["BAR"][1] <= jan_brix_ratio >= quality["BAR"][0] 
    # model.Brix_Min = Constraint(model.MONTHS, rule=brix)

    # Better way to define constraints for all quality parameters and months using a function and rule
    def quality_constraint(model, m, q):
        quality_cols = {
            "BAR": 7,
            "ACID": 8,
            "ASTRINGENCY": 9,
            "COLOR": 10,
        }
        ratio = sum(model.x[s, m] * suppliers_df.iloc[s - list(model.SUPPLIERS)[0], quality_cols[q]] for s in model.SUPPLIERS)
        '''
        Pyomo tuple syntax
        (a, expr, b)
        Pyomo interprets this as `a <= expr <= b`
        '''
        return (quality[q][0] * demand[m],
                 ratio,
                  quality[q][1] * demand[m]
                  )
        
    model.Quality_Constraint = Constraint(model.MONTHS, ["BAR", "ACID", "ASTRINGENCY", "COLOR"], rule=quality_constraint)

suppliers = load_supplier_data(df)
demand, Valencia_req, quality = load_demand_and_quality_constraints(df)

build_basic_model(suppliers, demand, Valencia_req, quality)