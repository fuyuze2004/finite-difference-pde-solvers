# finite-difference-pde-solvers
A numerical analysis suite implementing finite difference schemes, ghost cell boundary treatments, and stable SBP-SAT operators for linear hyperbolic PDEs, complete with convergence and stability benchmarks.

| Project | Boundary Condition | Schemes Tested | Key Finding |
|---|---|---|---|
| P1 | Periodic | Upwind, Crank–Nicholson, Lax–Wendroff, Leapfrog, FTCDS, Lax–Friedrichs, RK2CD2, RK4CD4 | FTCDS unconditionally unstable; RK4CD4 achieves order 4 with CFL limit ~2 |
| P2 | Inflow (ghost cells / one-sided) | Lax–Friedrichs, RK4CDS (ghost cell), RK4CDS (one-sided), RK4C4 (3 ghost cells) | RK4C4 degrades to order 2 due to boundary extrapolation instability |
| P3 | Inflow (SBP-SAT) | RK6 + SBP(2,1), SBP(4,2), SBP(6,3), SBP(4,4) | All four SBP-SAT schemes stable; resolves boundary instability seen in P2 |
