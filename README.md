# finite-difference-pde-solvers
A numerical analysis suite implementing finite difference schemes, ghost cell boundary treatments, and stable SBP-SAT operators for linear hyperbolic PDEs, complete with convergence and stability benchmarks.

### Core Components

| Project Suite | Scope & Focus | Implemented Methods |
| :--- | :--- | :--- |
| **`project-1-periodic-advection`** | Evaluates the spatial convergence orders and CFL stability conditions for finite difference schemes applied to an advection equation under periodic boundary conditions. | Upwind · Crank-Nicholson (CN) · Lax-Wendroff (LW) · Leapfrog (LF) · FTCDS · Lax-Friedrichs · RK2 · RK4 |
| **`project-2-ibvp-ghost-cells`** | Investigates standard finite difference methods for Initial Boundary Value Problems (IBVPs) involving linear hyperbolic PDEs, focusing closely on boundary treatments. | Lax-Friedrichs (1 ghost cell) · RK4CDS (1 ghost cell) · RK4CDS (one-sided difference formula) · RK4C4 (3 ghost cells) |
| **`project-3-sbp-sat-schemes`** | Explores Summation-by-Parts (SBP) operators combined with Simultaneous Approximation Terms (SAT) for energy-stable boundary closures, contrasting them against traditional ghost cells. | RK6 paired with SBP(2,1) · SBP(4,2) · SBP(6,3) · SBP(4,4) |
