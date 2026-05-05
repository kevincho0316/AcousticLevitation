Output folder guide
===================

Root
- summary.json: full machine-readable run config and results.
- README_outputs.txt: this guide.
- Field, trap, and z-scan outputs exclude z < 2.0 mm near the transducer plane.

Trap detection and filtering
- Primary condition: U_G is a 26-neighbor local minimum and lambda_min(H_U) > 0.
- Secondary filters: primary -> primary+p -> primary+p+lambda_min -> primary+p+lambda_min+depth.
- primary+p keeps the lowest local_p_ratio, where local_p_ratio = |p(candidate)| / percentile95(|p| within local_p_radius).
- lambda_min keeps the highest filter_fraction, and depth keeps the highest filter_fraction.
- refined/refined_candidates_*.csv refines each secondary candidate in a high-resolution local box and reports the refined minimum and Newton equilibrium estimate.
- Local-refine validation passes when the refined point is a 26-neighbor local minimum, lambda_min(H_U) > 0, the minimum is not on the refine-box edge, and the Newton equilibrium estimate remains inside the refine box.
- newton_offset_norm = ||-H_U^-1 grad(U_G)|| is a diagnostic displacement estimate, not a hard cutoff.
- final_candidates_*.csv and final_candidates_3d_*.png contain only candidates passing the local-refine validation checks.
- Boundary ties are preserved when p, local_p_ratio, U_G, grad_norm, depth, and lambda_min match within the physical tie tolerance.
- well_depth is estimated as min(U_G on a local Chebyshev shell) - U_G(candidate).
- Hyperparameters: filter_fraction = 0.5, depth_radius_cells = 2, local_p_radius = 0.0042875 m, local_p_reference_percentile = 95, local_p_min_samples = 100, local_refine_radius = 0.00107187 m, field_workers = 12, local_refine_grid_size = 101, local_refine_workers = 12, physical_tie_rtol = 1e-07, physical_tie_atol = 1e-12.

01_hardware_phase/
- hardware_trap.v: generated Verilog for the DE0-Nano output phase pattern.
- phase_table.csv: Tx position, path length, ideal phase, quantized tick, and phase error.
- quantized_ticks.txt: compact Tx01..Tx25 tick list.
- phases_ideal.png / phases_quantized.png: 5x5 phase maps.

02_trap_candidates/
- final_candidates_*.csv: final trusted trap candidates.
- coarse/: non-refined primary and secondary-stage candidate CSVs.
- refined/: high-resolution local-refine diagnostic CSVs.
- axis/: center-axis z scan diagnostic CSVs.

03_figures/field/
- field_slices_*.png: three pressure-field slices around the selected well.
- central_xz_sim_coords_*.png: paper-style central XZ normalized |p| view.
- z_scan_focus_xy.png: axial |p| scan at the phase-focus x,y position.

03_figures/traps_3d/
- primary_candidates_3d_*.png: 3D view of all primary trap candidates.
- primary+p*_3d_*.png: 3D view after each selected filter.
- final_candidates_3d_*.png: 3D view of final trusted refined candidates.

*_ideal uses continuous ideal phases. *_quantized uses FPGA tick-quantized phases.
