# finite-difference-pde-solvers
A numerical analysis suite implementing finite difference schemes, ghost cell boundary treatments, and stable SBP-SAT operators for linear hyperbolic PDEs, complete with convergence and stability benchmarks.

### 🛠️ Core Components

| Project Suite | Scope & Focus | Implemented Methods |
| :--- | :--- | :--- |
| **`📁 project-1-periodic-advection`** | [cite_start]Evaluates the spatial convergence orders and CFL stability conditions for finite difference schemes applied to an advection equation under periodic boundary conditions[cite: 4, 5]. | [cite_start]<ul><li>Upwind [cite: 15][cite_start]</li><li>Crank-Nicholson (CN) [cite: 35][cite_start]</li><li>Lax-Wendroff (LW) [cite: 62][cite_start]</li><li>Leapfrog (LF) [cite: 85][cite_start]</li><li>FTCDS [cite: 111][cite_start]</li><li>Lax-Friedrichs [cite: 129][cite_start]</li><li>RK2 [cite: 148][cite_start]</li><li>RK4 [cite: 172]</li></ul> |
| **`📁 project-2-ibvp-ghost-cells`** | [cite_start]Investigates standard finite difference methods for Initial Boundary Value Problems (IBVPs) involving linear hyperbolic PDEs, focusing closely on boundary treatments[cite: 856, 857]. | [cite_start]<ul><li>Lax-Friedrichs (1 ghost cell) [cite: 858][cite_start]</li><li>RK4CDS (1 ghost cell) [cite: 859][cite_start]</li><li>RK4CDS (one-sided difference formula) [cite: 860][cite_start]</li><li>RK4C4 (3 ghost cells) [cite: 861]</li></ul> |
| **`📁 project-3-sbp-sat-schemes`** | [cite_start]Explores Summation-by-Parts (SBP) operators combined with Simultaneous Approximation Terms (SAT) for energy-stable boundary closures, contrasting them against traditional ghost cells[cite: 1216]. | [cite_start]<ul><li>6th-Order Runge-Kutta (RK6) paired with [cite: 1217][cite_start]:</li><li>SBP(2,1) [cite: 1217][cite_start]</li><li>SBP(4,2) [cite: 1217][cite_start]</li><li>SBP(6,3) [cite: 1217][cite_start]</li><li>SBP(4,4) [cite: 1217]</li></ul> |
