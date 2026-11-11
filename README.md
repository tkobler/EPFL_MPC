# MPC_EPFL: Rocket Landing with Model Predictive Control

Course project for **Model Predictive Control** at EPFL (2025). The goal is to design
MPC controllers that steer and land a simulated rocket, step by step: linear MPC first,
then landing with robust MPC, then nonlinear MPC (NMPC).

## What was done

The work is split into deliverables, one folder each. Each folder holds a Jupyter notebook
and the controller code it uses.

| Step | Folder | Content |
|------|--------|---------|
| Linear MPC | [Deliverable_3_1](Deliverable_3_1/) | Linear MPC for the rocket velocities and roll, simulated on the linearised model |
| | [Deliverable_3_2](Deliverable_3_2/) | Same controllers, run with a target state (including an angle) |
| | [Deliverable_3_3](Deliverable_3_3/) | Linear MPC combined with the position controller from `PIControl` |
| | [Deliverable_4_1](Deliverable_4_1/) | Velocity MPC with invariant sets, plus position control |
| | [Deliverable_5_1](Deliverable_5_1/), [Deliverable_5_2](Deliverable_5_2/) | Velocity MPC with disturbance estimation, simulated on the nonlinear rocket with a mass different from the nominal model (5_1: lighter, no fuel consumption; 5_2: heavier, with fuel consumption) |
| Landing | [Deliverable_6_1&2](Deliverable_6_1%262/) | Robust (tube) MPC for landing, with minimal robust invariant sets; notebooks 6_1 and 6_2 |
| NMPC | [Deliverable_7](Deliverable_7/) | Nonlinear MPC for landing ([`nmpc_land.py`](Deliverable_7/nmpc_land.py), [notebook](Deliverable_7/Deliverable_7_1.ipynb)) |
| Other | [PIControl](PIControl/) | PI position controller used with the velocity MPC |
| | [Todo_1](Todo_1/), [Todo_2](Todo_2/) | Warm-up notebooks (Todo_1 simulates the nonlinear rocket) |

The nonlinear landing simulation and animation are in
[Deliverable_7/Deliverable_7_1.ipynb](Deliverable_7/Deliverable_7_1.ipynb)
(run it to see the rocket landing animation and the state/input plots).

The details, design choices and results are in the
[project report (PDF)](MPC_Project_Report_GroupBJ.pdf).

## Provided by the course

These files come from the EPFL course and are not our work:

- [`src/`](src/): rocket model, parameters and visualisation
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
to `sys.path` to find `src/`, `rocket.yaml` and the 3D models. Most notebooks import their
controller code (`LinearMPC`, `LandMPC`, `nmpc_land`) from that same folder. The exception is
`Deliverable_5_1` and `Deliverable_5_2`, which import it from the repository root through the
package path (`from Deliverable_5_1.LinearMPC.MPCVelControl import ...`); they still work
when launched from their own folder, because the parent directory (the repository root) is
on `sys.path`. Jupyter normally sets the working directory by default when you open the
notebook from the file browser. The notebooks also use the
`ipympl` widget backend (`%matplotlib widget`).

## Team

Noé Syfrig, Marwane Mroueh, Tim Kobler

## License

MIT, see [LICENSE](LICENSE). The files listed under "Provided by the course" come from the
EPFL course and are not covered by this license.
