import pandas as pd, numpy as np, math
import gurobipy as gp
from gurobipy import GRB, quicksum

# basic data
INCOME_THRESHOLD = 60000.0
EMPLOYMENT_THRESHOLD = 0.60
EMP_RATE_IN_PERCENT = False

# new facility menu
FACILITY_MENU = {
    "S": {"cap_total": 100, "cap_0_5": 50,  "cost_new":  65000},
    "M": {"cap_total": 200, "cap_0_5": 100, "cost_new":  95000},
    "L": {"cap_total": 400, "cap_0_5": 200, "cost_new": 115000},
}

COST_PER_NEW_0_5 = 100.0 # equipment cost per new 0–5 slot (applies to both new & expansions)

MIN_DIST_MILES = 0.06

# tiered expansion unit-cost parameters:
TIERS = {
    1: {"low": 0.00, "high": 0.10, "alpha":  200},   # 0–10%
    2: {"low": 0.10, "high": 0.15, "alpha":  400},   # 10–15%
    3: {"low": 0.15, "high": 0.20, "alpha": 1000},   # 15–20%
}

# help functions
# csv loading help function
def read_csv_robust(path, **kw):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1","iso-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip", low_memory=False, **kw)

# normalization of zip
def norm_zip(s):
    return str(s).strip().split(".")[0].zfill(5)

# looking for columns help
def find_col(df, candidates, required=True):
    c = next((c for c in df.columns if c.lower().strip().replace(" ","_") in
              [x.lower().strip().replace(" ","_") for x in candidates]), None)
    if required and c is None:
        raise ValueError(f"None of {candidates} found in columns: {list(df.columns)}")
    return c

# calculation of distance, given the coordination of two locations
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# data extract and clean

# Extract zip and population
pop = read_csv_robust("population.csv")
zc = "zipcode"
col_u5    = "-5"
col_59    = "5-9"
col_10_14 = "10-14"
# extract zip
pop.rename(columns={zc: "zip"}, inplace=True)
pop["zip"] = pop["zip"].map(norm_zip)
# extract 0-5, 5-9, 10-14
# estimate 0-12 as p05+p59+0.6*p1014
pop = pop.dropna(subset=["zip"]).copy()
p05   = pd.to_numeric(pop[col_u5],    errors="coerce").fillna(0.0)
p59   = pd.to_numeric(pop[col_59],    errors="coerce").fillna(0.0)
p1014 = pd.to_numeric(pop[col_10_14], errors="coerce").fillna(0.0)
pop["P_05"]  = p05
pop["P_all"] = p05 + p59 + 0.6 * p1014

# Extract employment rate
emp = read_csv_robust("employment_rate.csv")
zc = "zipcode"
emp.rename(columns={zc:"zip"}, inplace=True)
emp["zip"] = emp["zip"].map(norm_zip)
col_emp = "employment rate"
emp["emp_rate"] = emp[col_emp].astype(float)

# extract avg income
inc = read_csv_robust("avg_individual_income.csv")
zc = ("ZIP code")
inc.rename(columns={zc:"zip"}, inplace=True)
inc["zip"] = inc["zip"].map(norm_zip)
col_inc = "average income"
inc["avg_income"] = inc[col_inc].astype(float)

# merge according to zip level
zip_df = (pop[["zip","P_all","P_05"]]
          .merge(emp[["zip","emp_rate"]], on="zip", how="left")
          .merge(inc[["zip","avg_income"]], on="zip", how="left"))
zip_df.fillna({"emp_rate":0.0, "avg_income":1e9}, inplace=True)

# determine high demand area
zip_df["is_high_demand"] = ((zip_df["emp_rate"] >= EMPLOYMENT_THRESHOLD) |
                            (zip_df["avg_income"] <= INCOME_THRESHOLD)).astype(int)
# determine relevant requirement for desert shreshold
zip_df["tau"] = np.where(zip_df["is_high_demand"].eq(1), 0.5, 1/3)

# extract existing facilities data
fac = read_csv_robust("child_care_regulated.csv")
zc = "zip_code"
fac.rename(columns={zc:"zip"}, inplace=True)
fac["zip"] = fac["zip"].map(norm_zip)

col_id  = "facility_id"
col_tot = "total_capacity"
col_inf = "infant_capacity"
col_tod = "toddler_capacity"
lat_col = "latitude"
lon_col = "longitude"

fac = fac.dropna(subset=[col_id, "zip", col_tot]).copy()
fac["facility_id"] = fac[col_id].astype(str)
fac["C_total"] = fac[col_tot].astype(float)
fac["C_05"] = fac[[col_inf, col_tod]].astype(float).sum(axis=1) # 0-5 capacity = infant + toddler
fac["latitude"]  = pd.to_numeric(fac[lat_col], errors="coerce")
fac["longitude"] = pd.to_numeric(fac[lon_col], errors="coerce")

# aggregate current capacity per ZIP
caps = fac.groupby("zip", as_index=False).agg(
    S_zip=("C_total","sum"),
    S05_zip=("C_05","sum"),
    n_facilities=( "facility_id","count")
)
zip_df = zip_df.merge(caps, on="zip", how="left").fillna({"S_zip":0.0, "S05_zip":0.0})

# candidate locations for NEW
cand = read_csv_robust("potential_locations.csv")
zc = "zipcode"
cand.rename(columns={zc:"zip"}, inplace=True)
cand["zip"] = cand["zip"].map(norm_zip)
latc = "latitude"
lonc = "longitude"
cand["latitude"]  = pd.to_numeric(cand[latc], errors="coerce")
cand["longitude"] = pd.to_numeric(cand[lonc], errors="coerce")
cand = cand.dropna(subset=["zip","latitude","longitude"]).copy()
# define location ID
cand = cand.sort_values(["zip", "latitude", "longitude"]).reset_index(drop=True)
cand["cand_id"] = cand.index.astype(str)

# process sets
Z = zip_df["zip"].tolist() # define all ZIP
J = list(FACILITY_MENU.keys()) # define S/M/L facility type
F = fac["facility_id"].tolist() # define facility ID
I = cand["cand_id"].tolist() # define location ID

cap_total = {j: FACILITY_MENU[j]["cap_total"] for j in J} # total capacity for each facility type
cap_05    = {j: FACILITY_MENU[j]["cap_0_5"]   for j in J} # 0-5 capacity for each facility type
cost_new  = {j: FACILITY_MENU[j]["cost_new"]  for j in J} # cost for each type of facility

fac_idx  = fac.set_index("facility_id")
C_total  = fac_idx["C_total"].astype(float).to_dict() # total capacity for each current facility
C_05     = fac_idx["C_05"].astype(float).to_dict() # 0-5 capacity for each current facility
zip_of_f = fac_idx["zip"].astype(str).to_dict() # ZIP code for current each facility

zip_idx = zip_df.set_index("zip")
S_zip   = zip_idx["S_zip"].astype(float).to_dict() # current capacity in area Zip
S05_zip = zip_idx["S05_zip"].astype(float).to_dict() # 0-5 capacity in area Zip
P_all   = zip_idx["P_all"].astype(float).to_dict() # 0-12 demand in area zip
P_05    = zip_idx["P_05"].astype(float).to_dict() # 0-5 demand in area zip
tau     = zip_idx["tau"].astype(float).to_dict() # determination of care desert in each area

# mapping by ZIP
F_by_zip = fac.groupby("zip")["facility_id"].apply(list).to_dict()
I_by_zip = cand.groupby("zip")["cand_id"].apply(list).to_dict()

# Determine location within distance requirement
F_pos = [f for f, n in C_total.items() if n > 0.0]
F_by_zip_pos = {z: [f for f in F_by_zip.get(z, []) if C_total.get(f, 0.0) > 0.0] for z in Z}

# identify pairs of locations that does not satisfy the minimum distance constraint
bad_pairs = []
for z in Z:
    # extract all facility within each zip area
    Iz = I_by_zip[z]
    if len(Iz) <= 1: # if no. of facility less than one, process to another facility
        continue
    # identify corresponding facility id and location
    sub = cand.loc[cand["cand_id"].isin(Iz), ["cand_id","latitude","longitude"]]
    arr = sub.to_numpy()

    for a in range(len(arr)):
        for b in range(a+1, len(arr)):
            i, lati, loni = arr[a]
            k, latk, lonk = arr[b]
            # using haversine_miles function to calculate distance between two
            d = haversine_miles(lati, loni, latk, lonk)
            # if doesn't satisfy minimum distance, identify it as bad pairs
            if d < MIN_DIST_MILES:
                bad_pairs.append((str(i), str(k)))

# Apart from that, we should also determine whether a pair of potential location and existing location satisfy the distance requirement

allow_cand = {i: 1 for i in I}
for z in Z:
    # extract the current location and potential in each zip area
    Iz = I_by_zip.get(z, [])
    Fz = F_by_zip.get(z, [])
    if not Iz or not Fz:
        continue
    Ez = fac.loc[fac["facility_id"].isin(Fz) & fac["latitude"].notna() & fac["longitude"].notna(),
                 ["facility_id","latitude","longitude"]]
    if Ez.empty:
        continue

    # determine whether the potential location satisfy the distance requirement while comparing with current faciltiy location
    for i in Iz:
        ci = cand.loc[cand["cand_id"]==i, ["latitude","longitude"]].iloc[0]
        lat_i, lon_i = float(ci.latitude), float(ci.longitude)
        for rr in Ez.itertuples():
            d = haversine_miles(lat_i, lon_i, float(rr.latitude), float(rr.longitude))
            if d < MIN_DIST_MILES:
                allow_cand[i] = 0
                break

F_by_zip_raw = fac.groupby("zip")["facility_id"].apply(list).to_dict()
I_by_zip_raw = cand.groupby("zip")["cand_id"].apply(list).to_dict()

F_by_zip = {z: F_by_zip_raw.get(z, []) for z in Z}
I_by_zip = {z: I_by_zip_raw.get(z, []) for z in Z}

F_pos  = [f for f in F if C_total.get(f, 0.0) > 0.0]
F_zero = [f for f in F if C_total.get(f, 0.0) <= 0.0]

F_by_zip_pos = {z: [f for f in F_by_zip.get(z, []) if C_total.get(f, 0.0) > 0.0] for z in Z}

if "allow_cand" not in locals():
    allow_cand = {i: 1 for i in I}

m = gp.Model("Q2_Realistic_Expansion_and_Location")

# Decision Variable
# New facility selection & 0–5 allocation
# DV1: whether to build new facility at location i, with total capacity j
x = {(i,j): m.addVar(vtype=GRB.BINARY, ub=allow_cand[i], name=f"x[{i},{j}]") for i in I for j in J}
# DV2: the amount allocated to 0-5 slots in the new facility at location i, with total capacity j
u = {(i,j): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"u05[{i},{j}]") for i in I for j in J}

# Expansion on existing facility
# DV3: the expansion on the current facility f within tier k (expansion of 0-10%, 10%-15%, 15%-20%）
y = {(f,k): m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"y[{f},T{k}]") for f in F_pos for k in TIERS}
# DV4: whether to expand on the current facility f within tier k
b = {(f,k): m.addVar(vtype=GRB.BINARY, name=f"b[{f},T{k}]") for f in F_pos for k in TIERS}
# DV5: the expansion on current facility f that is allocated to 0-5 slot
v = {f: m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"v05[{f}]") for f in F_pos}

# Restrictions
# a) whether the number of 0-5 slots in a new faciltiy is within the limit
for i in I:
    m.addConstr(quicksum(x[i,j] for j in J) <= 1, name=f"one_size[{i}]")
    for j in J:
        m.addConstr(u[i,j] <= cap_05[j] * x[i,j], name=f"cap05_new[{i},{j}]")

# b) whether the expansion is within the particular tiers
for f in F_pos:
    nf = C_total[f]
    m.addConstr(quicksum(b[f,k] for k in TIERS) <= 1, name=f"one_tier[{f}]")
    for k, par in TIERS.items():
        low, high = par["low"], par["high"]
        m.addConstr(y[f,k] <= high * nf * b[f,k], name=f"ub_tier[{f},T{k}]")
        m.addConstr(y[f,k] >= low  * nf * b[f,k], name=f"lb_tier[{f},T{k}]")
    m.addConstr(v[f] <= quicksum(y[f,k] for k in TIERS), name=f"cap05_exp[{f}]")

# c) whether the new facility is not within a bad pair
for (i,k) in bad_pairs:
    m.addConstr(quicksum(x[i,j] for j in J) + quicksum(x[k,j] for j in J) <= 1,
                name=f"spacing_pair[{i},{k}]")

# d) whether the new zip area is not a child care desert
for z in Z:
    add_new_total = quicksum(cap_total[j] * x[i,j] for i in I_by_zip[z] for j in J)
    add_exp_total = quicksum(y[f,k] for f in F_by_zip_pos[z] for k in TIERS)
    m.addConstr(S_zip[z] + add_new_total + add_exp_total >= tau[z] * P_all[z],
                name=f"desert[{z}]")

# e) whether the 0-5 slot requirement is satisfied within each zip area
for z in Z:
    add_new_05 = quicksum(u[i,j] for i in I_by_zip[z] for j in J)
    add_exp_05 = quicksum(v[f]    for f in F_by_zip_pos[z])
    m.addConstr(S05_zip[z] + add_new_05 + add_exp_05 >= (2.0/3.0) * P_05[z],
                name=f"age05[{z}]")

# Objective
# OBJ1: total cost of new facility
obj_new = quicksum(cost_new[j] * x[i,j] for i in I for j in J)
# OBJ2: total cost of expansion
obj_exp = quicksum(((20000.0 / C_total[f]) + TIERS[k]["alpha"]) * y[f,k] for f in F_pos for k in TIERS)
# OBJ3: total equipment cost of 0-5 slots
obj_05  = COST_PER_NEW_0_5 * (quicksum(u[i,j] for i in I for j in J) + quicksum(v[f] for f in F_pos))
# Objective function
m.setObjective(obj_new + obj_exp + obj_05, GRB.MINIMIZE)

# Solve
m.Params.OutputFlag = 1
m.Params.MIPGap = 0.01
m.optimize()

status_map = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
}
print("\nModel status:", status_map.get(m.Status, m.Status))

y_keys = set(y.keys())              # {(f,k), ...}
b_keys = set(b.keys()) if 'b' in globals() else set()
v_keys = set(v.keys())              # {f, ...}
x_keys = set(x.keys())
u_keys = set(u.keys())

y_facilities = {fk[0] for fk in y_keys}   # 真实建了 y 的设施集合
b_facilities = {fk[0] for fk in b_keys} if b_keys else set()

if m.Status == GRB.OPTIMAL:
    print(f"Optimal total funding (Q2): ${m.ObjVal:,.0f}")

    # summary in every zip
    rows = []
    for z in Z:
        Iz_all = I_by_zip.get(z, [])
        Fz_all = F_by_zip.get(z, [])

        Fz = [f for f in Fz_all if (f in y_facilities or f in v_keys)]

        new_total = sum(
            cap_total[j] * x[i, j].X
            for i in Iz_all for j in J
            if (i, j) in x_keys
        )

        exp_total = sum(
            y[f, k].X
            for f in Fz for k in TIERS
            if (f, k) in y_keys
        )

        new_05 = sum(
            u[i, j].X
            for i in Iz_all for j in J
            if (i, j) in u_keys
        )

        exp_05 = sum(
            v[f].X for f in Fz if f in v_keys
        )

        rows.append({
            "zip": z,
            "new_sites": int(round(sum(x[i, j].X for i in Iz_all for j in J if (i, j) in x_keys))),
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

    picked = []
    for i in I:

        sizes = [j for j in J if (i, j) in x_keys and x[i, j].X > 0.5]
        if sizes:

            z_i = cand.loc[cand["cand_id"] == i, "zip"]
            z_i = z_i.iloc[0] if not z_i.empty else None
            picked.append({
                "cand_id": i,
                "zip": z_i,
                "size": sizes[0],
                "cap_total": cap_total[sizes[0]],
                "cap_0_5_max": cap_05[sizes[0]],
                "u05_selected": sum(u[i, j].X for j in J if (i, j) in u_keys),
                "build_cost": cost_new[sizes[0]],
                "equip_cost_0_5": COST_PER_NEW_0_5 * sum(u[i, j].X for j in J if (i, j) in u_keys),
            })
    picked_df = pd.DataFrame(picked).sort_values(["zip", "cand_id"])
    if not picked_df.empty:
        print("\n=== New facilities selected (first 20) ===")
        print(picked_df.head(20).to_string(index=False))
    else:
        print("\nNo new facilities selected.")

    exp_rows = []

    fac_iter = sorted(y_facilities | v_keys)
    for f in fac_iter:

        chosen_tiers = [k for k in TIERS if (f, k) in b_keys and b[f, k].X > 0.5]
        exp_amt = sum(y[f, k].X for k in TIERS if (f, k) in y_keys)
        if chosen_tiers or exp_amt > 1e-6:
            tier = chosen_tiers[0] if chosen_tiers else None
            alpha = TIERS[tier]["alpha"] if tier is not None else 0.0
            nf = C_total.get(f, 0.0)
            unit_cost = (20000.0 / nf + alpha) if (nf > 0 and exp_amt > 0) else 0.0
            zf = zip_of_f.get(f, None)
            exp_rows.append({
                "facility_id": f,
                "zip": zf,
                "tier": tier,
                "expansion_slots": exp_amt,
                "expansion_0_5": v[f].X if f in v_keys else 0.0,
                "cap_before": nf,
                "unit_cost_tier": unit_cost,
                "exp_cost_total": unit_cost * exp_amt
            })
    exp_df = pd.DataFrame(exp_rows).sort_values(["zip", "facility_id"])
    if not exp_df.empty:
        print("\n=== Expanded existing facilities (first 20) ===")
        print(exp_df.head(20).to_string(index=False))

elif m.Status == GRB.INFEASIBLE:
    print("\nModel is infeasible. Computing IIS...")
    m.computeIIS()
    m.write("Q2_infeasible.ilp")
    print("IIS written to Q2_infeasible.ilp")
