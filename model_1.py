# ========================== #
#  STATEWIDE DESERT MIP (NYS)
# ========================== #
# Requires: gurobipy, pandas, numpy
# Files expected in working dir:
#   population.csv, employment_rate.csv, avg_individual_income.csv, child_care_regulated.csv

import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# ========================== #
#   CONFIG for your CSVs
# ========================== #

COLS = {
    # population.csv
    "pop_zip": "zipcode",        # ZIP code
    "pop_total": "Total",        
    "pop_under5": "-5",          
    # employment_rate.csv
    "emp_zip": "zipcode",
    "employment_rate": "employment rate",
    # avg_individual_income.csv
    "inc_zip": "ZIP code",
    "avg_income": "average income",
    # child_care_regulated.csv
    "fac_zip": "zip_code",
    "facility_id": "facility_id",
    "fac_total_capacity": "total_capacity",
    "fac_infant": "infant_capacity",
    "fac_toddler": "toddler_capacity",
    "fac_preschool": "preschool_capacity"
}

EMP_RATE_IN_PERCENT = False
INCOME_THRESHOLD = 60000.0
EMPLOYMENT_THRESHOLD = 0.60
DEFAULT_EXP_COST_PER_SLOT = 300.0  # no column for expansion cost → use default

# Facility menu (same)
FACILITY_MENU = {
    "S": {"cap_total": 100, "cap_0_5": 50,  "cost_new": 65000},
    "M": {"cap_total": 200, "cap_0_5": 100, "cost_new": 95000},
    "L": {"cap_total": 400, "cap_0_5": 200, "cost_new": 115000},
}
COST_PER_NEW_0_5 = 100.0

# ========================== #
#   LOAD + CLEAN DATA
# ========================== #
import pandas as pd
import numpy as np

# population
pop = pd.read_csv("population.csv", usecols=["zipcode", "Total", "-5"])
pop = pop.dropna(how="any")
pop["zipcode"] = pop["zipcode"].astype(str).str.zfill(5)
pop.rename(columns={"-5": "P_05", "Total": "TotalPop"}, inplace=True)
pop["P_all"] = pop["TotalPop"]  
#pop["P_all"] = pop["5-9"] + pop["10-14"] + pop["-5"]

# employment rate
emp = pd.read_csv("employment_rate.csv", usecols=["zipcode", "employment rate"])
emp = emp.dropna(how="any")
emp["zipcode"] = emp["zipcode"].astype(str).str.zfill(5)
emp.rename(columns={"employment rate": "emp_rate"}, inplace=True)
if EMP_RATE_IN_PERCENT:
    emp["emp_rate"] = emp["emp_rate"] / 100.0

# income
inc = pd.read_csv("avg_individual_income.csv", usecols=["ZIP code", "average income"])
inc = inc.dropna(how="any")
inc.rename(columns={"ZIP code": "zipcode", "average income": "avg_income"}, inplace=True)
inc["zipcode"] = inc["zipcode"].astype(str).str.zfill(5)

# merge by zipcode
zip_df = (
    pop.merge(emp, on="zipcode")
        .merge(inc, on="zipcode")
        .rename(columns={"zipcode": "zip"})
)

# classify demand level
zip_df["is_high_demand"] = ((zip_df["emp_rate"] >= EMPLOYMENT_THRESHOLD) |
                            (zip_df["avg_income"] <= INCOME_THRESHOLD)).astype(int)
zip_df["tau"] = np.where(zip_df["is_high_demand"] == 1, 0.5, 1/3)

# facilities
fac = pd.read_csv("child_care_regulated.csv",
                  usecols=["facility_id", "zip_code",
                           "total_capacity", "infant_capacity",
                           "toddler_capacity", "preschool_capacity"])
fac = fac.dropna(subset=["zip_code", "total_capacity"])
fac["zip_code"] = fac["zip_code"].astype(str).str.zfill(5)
fac.rename(columns={"zip_code": "zip"}, inplace=True)

# compute 0–5 capacity (infant + toddler + preschool)
fac["C_05"] = fac[["infant_capacity", "toddler_capacity", "preschool_capacity"]].sum(axis=1)
fac["C_total"] = fac["total_capacity"]
fac["exp_cost"] = DEFAULT_EXP_COST_PER_SLOT
fac["addMax"] = np.minimum(0.2 * fac["C_total"], 500)

# aggregate to ZIP
zip_caps = fac.groupby("zip", as_index=False).agg(
    S_zip=("C_total", "sum"),
    S05_zip=("C_05", "sum"),
    n_facilities=("facility_id", "count")
)
zip_df = zip_df.merge(zip_caps, on="zip", how="left").fillna({"S_zip": 0, "S05_zip": 0})

# ---------------------------
# 3) BUILD MIP
# ---------------------------

Z = list(zip_df["zip"].unique())               #ZIP
J = list(FACILITY_MENU.keys())                 
F = list(fac["facility_id"].unique())          

F_by_zip = {z: fac.loc[fac["zip"] == z, "facility_id"].tolist() for z in Z}

cap_total = {j: FACILITY_MENU[j]["cap_total"] for j in J}
cap_05    = {j: FACILITY_MENU[j]["cap_0_5"]   for j in J}
cost_new  = {j: FACILITY_MENU[j]["cost_new"]  for j in J}

C_total   = {row["facility_id"]: row["C_total"] for _, row in fac.iterrows()}
C_05      = {row["facility_id"]: row["C_05"]    for _, row in fac.iterrows()}
addMax    = {row["facility_id"]: min(0.2 * row["C_total"], 500)
             for _, row in fac.iterrows()}
expCost   = {row["facility_id"]: DEFAULT_EXP_COST_PER_SLOT
             for _, row in fac.iterrows()}  

S_zip   = {row["zip"]: float(row["S_zip"]) for _, row in zip_df.iterrows()}
S05_zip = {row["zip"]: float(row["S05_zip"]) for _, row in zip_df.iterrows()}
P_all   = {row["zip"]: float(row["P_all"]) for _, row in zip_df.iterrows()}
P_05    = {row["zip"]: float(row["P_05"]) for _, row in zip_df.iterrows()}
tau     = {row["zip"]: float(row["tau"]) for _, row in zip_df.iterrows()}

m = Model("NYS_Childcare_Deserts_Statewide")

# Decision variables
x = {(z,j): m.addVar(vtype=GRB.INTEGER, name=f"x[{z},{j}]") for z in Z for j in J}  # new facilities
u = {(z,j): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u05[{z},{j}]") for z in Z for j in J}  # new 0–5 slots
y = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y_exp[{f}]") for f in F}  # expansion slots per facility
v = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"v05[{f}]") for f in F}   # 0–5 portion of expansion

# Capacity consistency
for z in Z:
    for j in J:
        m.addConstr(u[z,j] <= cap_05[j] * x[z,j], name=f"cap05_new[{z},{j}]")

for f in F:
    m.addConstr(y[f] <= addMax[f], name=f"cap_exp[{f}]")
    m.addConstr(v[f] <= y[f],       name=f"cap05_exp[{f}]")

# Desert elimination per ZIP: total slots after > threshold
for z in Z:
    m.addConstr(
        S_zip[z] + quicksum(cap_total[j]*x[z,j] for j in J) + quicksum(y[f] for f in F_by_zip[z])
        >= tau[z] * P_all[z],
        name=f"desert[{z}]"
    )

# 0–5 rule per ZIP: ≥ 2/3 of P_05
for z in Z:
    m.addConstr(
        S05_zip[z] + quicksum(u[z,j] for j in J) + quicksum(v[f] for f in F_by_zip[z])
        >= (2.0/3.0) * P_05[z],
        name=f"age05[{z}]"
    )

# Objective: build + expansion + $100 per new 0–5 slot
obj = quicksum(cost_new[j] * x[z,j] for z in Z for j in J) \
    + quicksum(expCost[f] * y[f]     for f in F) \
    + COST_PER_NEW_0_5 * (quicksum(u[z,j] for z in Z for j in J) + quicksum(v[f] for f in F))

m.setObjective(obj, GRB.MINIMIZE)

# Solve
m.Params.OutputFlag = 1
m.optimize()

# ---------------------------
# 4) REPORT
# ---------------------------
status_map = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
}
print("\nModel status:", status_map.get(m.Status, m.Status))

if m.Status == GRB.OPTIMAL:
    print(f"Optimal total funding: ${m.ObjVal:,.0f}")

    # Summaries per ZIP
    rows = []
    for z in Z:
        new_total = sum(cap_total[j]*x[z,j].X for j in J)
        exp_total = sum(y[f].X for f in F_by_zip[z])
        new_05    = sum(u[z,j].X for j in J)
        exp_05    = sum(v[f].X for f in F_by_zip[z])

        rows.append({
            "zip": z,
            "new_S": int(round(x[z,'S'].X)) if ('S' in J) else 0,
            "new_M": int(round(x[z,'M'].X)) if ('M' in J) else 0,
            "new_L": int(round(x[z,'L'].X)) if ('L' in J) else 0,
            "add_slots_new_total": new_total,
            "add_slots_exp_total": exp_total,
            "add_slots_0_5_total": new_05 + exp_05,
            "post_total_slots": S_zip[z] + new_total + exp_total,
            "needed_total_slots": tau[z] * P_all[z],
            "post_0_5_slots": S05_zip[z] + new_05 + exp_05,
            "needed_0_5_slots": (2.0/3.0) * P_05[z],
        })
    out = pd.DataFrame(rows).sort_values("zip")
    print("\n=== ZIP summary (first 20) ===")
    print(out.head(20).to_string(index=False))

    # Facility-level expansion (top 20 nonzero)
    fac_rows = []
    for f in F:
        if y[f].X > 1e-6:
            fac_rows.append({
                "facility_id": f,
                "zip": fac.loc[fac["facility_id"] == f, "zip"].iloc[0],
                "expansion_slots": y[f].X,
                "expansion_0_5": v[f].X,
                "cap_total_before": C_total[f],
                "cap_05_before": C_05[f],
                "addMax": addMax[f],
                "exp_cost_per_slot": expCost[f],
                "exp_cost_total": expCost[f]*y[f].X
            })
    fac_out = pd.DataFrame(fac_rows).sort_values(["zip","facility_id"])
    if not fac_out.empty:
        print("\n=== Facilities expanded (first 20) ===")
        print(fac_out.head(20).to_string(index=False))
    else:
        print("\nNo facility expansions chosen; all coverage achieved via new builds.")

elif m.Status == GRB.INFEASIBLE:
    print("\nModel is infeasible. Computing IIS...")
    m.computeIIS()
    m.write("infeasible.ilp")
    print("IIS written to infeasible.ilp (constraints/vars that conflict).")
