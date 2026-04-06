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
    # Binary indicator: 1 if supplier s is used in month m, 0 otherwise
    model.y = Var(model.SUPPLIERS, model.MONTHS, domain=Binary)
    model.r = Var(model.SUPPLIERS, model.MONTHS, domain=Binary) # Binary variable for reducer usage
    model.z = Var(model.SUPPLIERS, model.MONTHS, domain=NonNegativeReals) # Binary variable z[s, m] = r[s, m] * x[s, m]. 
    # So z stands for the quantity of the order that is affected by the reducer.

    # Parameters
    available = suppliers_df.iloc[:, 6].tolist()
    # brix_acid = suppliers_df.iloc[:, 7].tolist()
    # acid = suppliers_df.iloc[:, 8].tolist()
    # astringency = suppliers_df.iloc[:, 9].tolist()
    # color = suppliers_df.iloc[:, 10].tolist()
    price = suppliers_df.iloc[:, 11].tolist()
    shipping = suppliers_df.iloc[:, 12].tolist()

    # Objective function: minimize total cost = sum of (quantity from supplier * (price + shipping))
    cost = 0
    for s in model.SUPPLIERS:
        for m in model.MONTHS:
            idx = s - list(model.SUPPLIERS)[0]
            cost += model.x[s, m] * (price[idx] + shipping[idx]) + model.z[s, m] * 20 # The cost of using the reducer is $20 per order reduced
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
    # model.Demand_Jan_Alt = Constraint(expr=sum(model.x[s, "January"] for s in model.SUPPLIERS) >= demand["January"])
    # model.Demand_Feb_Alt = Constraint(expr=sum(model.x[s, "February"] for s in model.SUPPLIERS) >= demand["February"])
    # model.Demand_Mar_Alt = Constraint(expr=sum(model.x[s, "March"] for s in model.SUPPLIERS) >= demand["March"])

    # Constraints written with rules and functions
    ### Each month must meet a specific demand. Final
    def demand_function(model, m):
        return sum(model.x[s, m] for s in model.SUPPLIERS) == demand[m]
    model.demand_constraint = Constraint(model.MONTHS, rule=demand_function)

    ### Each supplier has a maximum available quantity that cannot be exceeded
    def available_constraint(model, s):
        idx = s - list(model.SUPPLIERS)[0]
        return sum(model.x[s, m] for m in model.MONTHS) <= available[idx]
    model.Available_constraint = Constraint(model.SUPPLIERS, rule=available_constraint)

    ### Valencia content constraints. Valencia orders for each month must make up at least 40% of the total blend.
    def valencia_constraint(model, m):
        # Supplier at row 7 is Valencia
        return model.x[7, m] >= Valencia_req[m]
    model.Valencia_constraint = Constraint(model.MONTHS, rule=valencia_constraint)

    ## Constraint: max no of suppliers must be 4.
    def max_suppliers_constraint(model, m):
        # Count only suppliers with y[s, m] = 1 (actually used)
        return sum(model.y[s, m] for s in model.SUPPLIERS) <= 4
    model.Max_Suppliers_Constraint = Constraint(model.MONTHS, rule=max_suppliers_constraint)

    # Big-M constraint to link binary variables to continuous variables
    # If x[s, m] > 0, then y[s, m] must be 1
    # If y[s, m] = 0, then x[s, m] must be 0
    def link_binary_constraint(model, s, m):
        idx = s - list(model.SUPPLIERS)[0]
        # x[s, m] <= available[idx] * y[s, m]
        # If y = 0, x is forced to 0; if y = 1, x can be up to available qty
        return model.x[s, m] <= available[idx] * model.y[s, m]
        '''
        This ensures:
        if y[s,m] = 0 → x[s,m] = 0
        if x[s,m] > 0 → y[s,m] = 1
        '''
    model.Link_Binary_Constraint = Constraint(model.SUPPLIERS, model.MONTHS, rule=link_binary_constraint)

    def reducer_bound_constraint_upper1(model, s, m):
        return model.z[s, m] <= model.x[s, m]
    model.Reducer_upper1 = Constraint(model.SUPPLIERS, model.MONTHS, rule=reducer_bound_constraint_upper1)

    def reducer_bound_constraint_upper2(model, s, m):
        idx = s - list(model.SUPPLIERS)[0]
        return model.z[s, m] <= available[idx] * model.r[s, m]
    model.Reducer_upper2 = Constraint(model.SUPPLIERS, model.MONTHS, rule=reducer_bound_constraint_upper2)

    def reducer_bound_constraint_lower(model, s, m):
        return model.z[s, m] >= model.x[s, m] - available[s - list(model.SUPPLIERS)[0]] * (1 - model.r[s, m])
    model.Reducer_lower = Constraint(model.SUPPLIERS, model.MONTHS, rule=reducer_bound_constraint_lower)


    ## Quality constraints
    ### Draft for January Brix/Acid ratio constraint
    # jan_brix_acid = 0
    # for s in model.SUPPLIERS:
    #     for m in model.MONTHS:
    #         idx = s - list(model.SUPPLIERS)[0]
    #         jan_brix_acid += model.x[s, m] * suppliers_df.iloc[idx, 7] 
    # jan_brix_acid_ratio = jan_brix_acid / demand["January"]
    # model.Jan_Brix_Min = Constraint(expr=jan_brix_acid_ratio >= quality["BAR"][0])
    # model.Jan_Brix_Max = Constraint(expr=jan_brix_acid_ratio <= quality["BAR"][1])

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
        quality_cols_reduced = {
            "BAR": [7, 1.25],
            "ACID": [8, 0.8],
            "ASTRINGENCY": [9, 1.0],
            "COLOR": [10, 1.0],
        }
        ratio_reduced = sum(model.z[s, m] * suppliers_df.iloc[s - list(model.SUPPLIERS)[0], quality_cols_reduced[q][0]] * quality_cols_reduced[q][1] for s in model.SUPPLIERS)
        ratio_not_reduced = sum((model.x[s, m] - model.z[s, m]) * suppliers_df.iloc[s - list(model.SUPPLIERS)[0], quality_cols[q]] for s in model.SUPPLIERS)
        ratio = ratio_reduced + ratio_not_reduced
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

    return model

# Creating the four args to build the basic model
suppliers = load_supplier_data(df)
demand, Valencia_req, quality = load_demand_and_quality_constraints(df)
# Build the basic model
model = build_basic_model(suppliers, demand, Valencia_req, quality)
# Get the solver and solve the model
# Pass the model to the solver
solver = SolverFactory("glpk")
print(solver.available())
results = solver.solve(model)

# Check solver status
if (results.solver.status == 'ok') and \
   (results.solver.termination_condition == TerminationCondition.optimal):
    print("✅ Optimal solution found")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("⚠️ Model is infeasible")
elif results.solver.termination_condition == TerminationCondition.unbounded:
    print("⚠️ Model is unbounded")
else:
    print("⚠️ Solver did not succeed")
    print("Status:", results.solver.status)
    print("Termination:", results.solver.termination_condition)

# Extract solution if available
if results.solver.termination_condition == TerminationCondition.optimal:
    for s in model.SUPPLIERS:
        for m in model.MONTHS:
            print(f"{s}, {m}: {model.x[s, m].value}")
            print(f"Reducer used: {model.r[s, m].value}, Quantity reduced: {model.z[s, m].value}")
    print("Total Cost:", model.OBJ())

