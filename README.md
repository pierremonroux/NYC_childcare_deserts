# NYC_childcare_deserts
# Child Care Deserts Optimization

This repository contains the implementation and report for an optimization project aiming to eliminate child care deserts across New York State. It includes two mixed-integer linear programming (MILP) models that determine the most cost-effective combination of facility expansions and new constructions.

_Data files are not included in this repository due to licensing restrictions._

## Contents
- **report.pdf** — Full project report describing the models, assumptions, and results.
- **model_1/** — Python/Gurobi implementation of the initial statewide budgeting MILP.
- **model_2/** — Python/Gurobi implementation of the refined model with realistic expansion costs and facility-spacing constraints.

## Requirements
- Python 3
- `gurobipy`
- `pandas`
- `numpy`

## Notes
The code expects the original datasets (population, income, employment, facility data) to be placed in the appropriate directories. These files are not distributed in this repository. Please supply them yourself to run the models.

