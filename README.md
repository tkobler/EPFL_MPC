# MPC_EPFL: Rocket Landing with Model Predictive Control

Course project for **Model Predictive Control** at EPFL (2025). The goal is to design
MPC controllers that steer and land a simulated rocket, step by step: linear MPC first,
then landing with robust MPC, then nonlinear MPC (NMPC).

Team: **Group BJ** (member names are on the cover page of the
[project report](MPC_Project_Report_GroupBJ.pdf)).

## What was done

The work is split into deliverables, one folder each. Each folder holds a Jupyter notebook
and the controller code it uses.

| Step | Folder | Content |
|------|--------|---------|
| Linear MPC | [Deliverable_3_1](Deliverable_3_1/) | Linear MPC for the rocket velocities and roll, simulated on the linearised model |
| | [Deliverable_3_2](Deliverable_3_2/) | Same controllers, run with a target state (including an angle) |
| | [Deliverable_3_3](Deliverable_3_3/) | Linear MPC combined with the position controller from `PIControl` |
| | [Deliverable_4_1](Deliverable_4_1/) | Velocity MPC with invariant sets, plus position control |
| | [Deliverable_5_1](Deliverable_5_1/), [Deliverable_5_2](Deliverable_5_2/) | Velocity MPC with disturbance estimation, simulated on the nonlinear rocket |
| Landing | [Deliverable_6_1&2](Deliverable_6_1%262/) | Robust (tube) MPC for landing, with minimal robust invariant sets; notebooks 6_1 and 6_2 |
| NMPC | [Deliverable_7](Deliverable_7/) | Nonlinear MPC for landing ([`nmpc_land.py`](Deliverable_7/nmpc_land.py), [notebook](Deliverable_7/Deliverable_7_1.ipynb)) |
| Other | [PIControl](PIControl/) | PI position controller used with the velocity MPC |
| | [Todo_1](Todo_1/), [Todo_2](Todo_2/) | Warm-up notebooks (Todo_1 simulates the nonlinear rocket) |

The nonlinear landing simulation and animation are in
[Deliverable_7/Deliverable_7_1.ipynb](Deliverable_7/Deliverable_7_1.ipynb)
(run it to see the rocket landing animation and the state/input plots).

The details, design choices and results are in the
[project report](MPC_Project_Report_GroupBJ.pdf).

## Provided by the course

These files come from the EPFL course and are not our work:

- [`src/`](src/): rocket model, parameters and visualisation
- [`LinearMPC_template/`](LinearMPC_template/) and [`LandMPC_template/`](LandMPC_template/): code templates
- [`EPFL_MPC_Course_Project_Description_2025.pdf`](EPFL_MPC_Course_Project_Description_2025.pdf): the project description
- `rocket.yaml`, `Simple_rocket.obj`, `Simple_rocket.mtl`, `Cartoon_rocket.obj`: rocket parameters and 3D models

## Technologies

Python, Jupyter, NumPy, SciPy, CasADi (optimisation, Ipopt), CVXPY, python-control,
mpt4py (polytopes and invariant sets), Matplotlib and PyVista (visualisation).

## How to run

Requires Python 3.10 or newer (`src/rocket.py` uses `match`).

```bash
pip install -r requirements.txt
jupyter notebook
```

Open a notebook and run it from **its own deliverable folder** (for example
`Deliverable_7/`): each notebook adds the parent directory of the current working directory
to `sys.path` to find `src/`, `rocket.yaml` and the 3D models, and imports its controller code
(`LinearMPC`, `LandMPC`, `nmpc_land`) from that same folder. Jupyter normally does this
by default when you open the notebook from the file browser. The notebooks also use the
`ipympl` widget backend (`%matplotlib widget`).

## License

No license file has been added to this repository yet.
