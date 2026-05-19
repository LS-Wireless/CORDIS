# CORDIS Configuration Parameter Reference

Generated automatically from `cordis/utils/config.py`.


## Topology

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `type` | str | `circle` | - | {circle|random|grid} | AP placement geometry |
| `ap_radius_m` | float | `650.0` | m | > 0 | Radius of the circle on which APs are placed (circle layout) |
| `ue_max_radius_m` | float | `1000.0` | m | > ue_min_radius_m | Outer radius of the annulus in which UEs are dropped |
| `ue_min_radius_m` | float | `35.0` | m | >= 0 | Inner radius of the UE placement annulus (minimum UE-origin distance) |
| `tg_max_radius_m` | float | `1000.0` | m | > tg_min_radius_m | Outer radius of the annulus in which targets are dropped |
| `tg_min_radius_m` | float | `35.0` | m | >= 0 | Inner radius of the target placement annulus |
| `n_ap` | int | `6` | - | >= 2 | Total number of APs (|A| = |A_t| + |A_r|) |
| `n_ue` | int | `4` | - | >= 1 | Number of single-antenna communication users (N_ue) |
| `n_targets` | int | `1` | - | >= 1 | Number of sensing targets (N_t) |
| `n_ant` | int | `10` | - | >= 1 | Number of antenna elements per AP (M_t = M_r = M) |
| `n_rf_chains` | int | `10` | - | [1, n_ant] | Number of RF chains per AP; limits simultaneous streams (N_RF <= M) |
| `array_type` | str | `UCA` | - | {ULA|UCA} | Antenna array geometry at every AP |
| `antenna_spacing_factor` | float | `0.5` | lambda | (0, 1] | Normalised inter-element spacing d/lambda |
| `n_sensing_rx` | int | `1` | - | [1, n_ap - 1] | Number of APs dedicated to sensing reception (|A_r|); remaining APs transmit |
| `h_ap_m` | float | `10.0` | m | > h_ue_m | AP antenna height above ground (used in 3GPP path loss) |
| `h_ue_m` | float | `1.5` | m | > 0 | UE antenna height above ground |
| `h_tg_m` | float | `1.5` | m | >= 0 | Target height above ground |
| `grid_n_rows` | int | `3` | - | >= 1 | Number of rows in the AP grid (grid layout only); total APs = rows x cols |
| `grid_n_cols` | int | `2` | - | >= 1 | Number of columns in the AP grid (grid layout only) |
| `grid_spacing_m` | float | `433.0` | m | > 0 | Distance between adjacent APs in the grid |

## Frequency & Waveform

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `carrier_freq_ghz` | float | `3.5` | GHz | [0.5, 100]  (3GPP TR 38.901 validity range) | Carrier frequency f_c; determines wavelength and path loss exponents |
| `bandwidth_mhz` | float | `20.0` | MHz | > 0 | Total system bandwidth B |
| `n_subcarriers` | int | `64` | - | power of 2, >= rb_n_subcarriers | Total number of OFDM subcarriers N_sc |
| `rb_n_subcarriers` | int | `12` | - | >= 1 | Number of subcarriers per resource block Q |
| `rb_n_symbols` | int | `14` | - | >= 1 | Number of OFDM symbols per resource block K |
| `subcarrier_spacing_khz` | float | `15.0` | kHz | > 0 | OFDM subcarrier spacing Delta_f (15 kHz = 5G NR numerology 0) |
| `cp_fraction` | float | `0.0714` | - | (0, 0.25] | Cyclic prefix duration as a fraction of the OFDM symbol period |

## Channel Model

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `model` | str | `3gpp_umi_rician` | - | {3gpp_umi_rician} | Channel model identifier string (reserved for future model selection) |
| `scenario` | str | `UMi` | - | {UMi|UMa|RMa} | 3GPP propagation scenario; controls path loss and LoS probability formulas |
| `environment` | str | `StreetCanyon` | - | - |  |
| `snr_db` | float | `20.0` | dB | any real (typical: 0 to 30) | Transmit SNR = P_max / sigma_n^2; sets the power budget for all APs |
| `noise_figure_db` | float | `7.0` | dB | >= 0 | Receiver noise figure NF added to thermal noise floor |
| `noise_temp_k` | float | `290.0` | K | > 0 (standard: 290) | Thermal noise reference temperature T_0 |
| `tau_f` | int | `200` | samples | > tau_p + tau_d | TDD frame size tau_f in channel uses; must satisfy tau_f >= tau_p + tau_d |
| `tau_p` | int | `10` | samples | >= 1 | Pilot sequence length tau_p; set >= n_ue to avoid pilot contamination |
| `tau_d` | int | `100` | samples | >= 1 | Number of downlink ISAC symbols per TDD frame (tau_d) |
| `pilot_power_db` | float | `120.0` | dB | any real; 80-130 dB for 3GPP UMi at 50-650 m; 10-30 dB only for no-path-loss toy models | Uplink pilot transmit SNR: 10 log10(P_p / sigma_n^2). The RECEIVED pilot SNR at the AP is SNR_rx = (P_p/sigma_n^2) x tau_p x beta_{au}. With 3GPP UMi path loss, beta << 1, so this must be set much higher than the data SNR to compensate. Rule of thumb: pilot_power_db = snr_db + |PL_dB|. 120 dB corresponds to P_p ~ 200 mW (23 dBm), which is physically realistic. |
| `shadow_fading_los_std_db` | float | `4.0` | dB | >= 0 | Shadow fading standard deviation sigma_SF for LoS links (3GPP UMi: 4 dB) |
| `shadow_fading_nlos_std_db` | float | `7.82` | dB | >= 0 | Shadow fading standard deviation sigma_SF for NLoS links (3GPP UMi: 7.82 dB) |
| `shadow_corr_distance_m` | float | `50.0` | m | > 0 (3GPP default: 50 m) | Shadow fading spatial decorrelation distance D_corr between UEs at a common AP |
| `rician_k_db_mean` | float | `9.0` | dB | any real | Mean Rician K-factor for LoS links (drawn from Gaussian; 3GPP UMi LoS: 9 dB) |
| `rician_k_db_std` | float | `5.0` | dB | >= 0 | Standard deviation of the Rician K-factor (3GPP UMi: 5 dB) |
| `as_azimuth_deg_std_los` | float | `5.0` | deg | >= 0 | Azimuth angular spread (ASD) sigma_phi for LoS links; used to build C_{au} |
| `as_azimuth_deg_std_nlos` | float | `22.5` | deg | >= 0 | Azimuth angular spread (ASD) sigma_phi for NLoS links |
| `as_elevation_deg_std_los` | float | `3.0` | deg | >= 0 | Elevation angular spread (ESD) sigma_theta for LoS links |
| `as_elevation_deg_std_nlos` | float | `7.0` | deg | >= 0 | Elevation angular spread (ESD) sigma_theta for NLoS links |
| `n_spatial_samples` | int | `2000` | - | >= 100 (typical: 1000-5000) | Monte Carlo samples for integrating the spatial correlation matrix C_{au}; higher = more accurate but slower |
| `shared_scatter_rank` | int | `0` | - | [0, n_ant) | Rank r of shared scattering subspace B_a (inter-user correlation); 0 = disabled |
| `shared_scatter_power` | float | `0.1` | - | [0, 1) | Fraction of NLoS power allocated to the shared scattering subspace B_a |
| `estimation_method` | str | `MMSE` | - | {MMSE|LS|perfect}  (perfect = genie-aided, zero estimation error) | Channel estimation algorithm at transmit APs |
| `mmwave_n_clusters` | int | `2` | - | >= 1 | Number of mmWave scattering clusters (active when carrier_freq_ghz > 6) |
| `mmwave_n_rays` | int | `10` | - | >= 1 | Number of rays per mmWave cluster |
| `mmwave_cluster_as_deg` | float | `10.0` | deg | > 0 | Per-cluster angular spread for mmWave channel |

## Sensing & Clutter

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `sigma_rcs_sq_db` | float | `-3.0` | dB | any real (typical: -10 to 10) | RCS power variance sigma_RCS^2 = E[|zeta_t|^2]; Swerling-I model |
| `swerling_model` | int | `1` | - | {0|1} | Target RCS fluctuation model: 0 = deterministic, 1 = Swerling-I (CN) |
| `rx_strategy` | str | `single_closest_centroid` | - | {single_closest_centroid|single_farthest_centroid|per_target_closest|assent_file|assent_live} | Receive AP selection strategy; determines which APs collect target echoes |
| `los_model` | str | `3gpp_umi` | - | {3gpp_umi|always|deterministic} | LoS availability model for target s_t; controls stochastic blockage |
| `los_probability_override` | Optional[float] | `null` | - | [0, 1] or null | Fixed P_LoS value for all targets when los_model='deterministic'; null = use model |
| `multi_static` | bool | `true` | - | {true|false} | Enable multi-static sensing (multiple TX/RX AP pairs per target) |
| `max_tx_aps_per_target` | int | `5` | - | [1, n_ap - 1] | Maximum number of transmit APs illuminating a single target K_tx |
| `max_rx_aps_per_target` | int | `3` | - | [1, n_sensing_rx] | Maximum number of receive APs processing echoes from a single target K_rx |
| `n_snapshots` | int | `20` | symbols | >= 1 | Number of slow-time STAP snapshots T; multiplicative temporal gain in SCNR |
| `sigma_clt` | Optional[float] | `null` | - | null OR >= 0 | OPTIONAL explicit override for sigma_clt.  If null (default), sigma_clt is derived from clutter_cnr_db; if set, this value is used directly (sigma_clt^2 = sigma_clt^2, ignoring CNR). |
| `clutter_cnr_db` | float | `-10.0` | dB | any real (typical: -20 to 0) | Clutter-to-noise ratio (CNR); sets sigma_clt^2 = 10^(CNR/10) * sigma_n^2.  Used only when sensing.sigma_clt is null (default). |
| `rho_clt_model` | str | `constant` | - | {constant|jakes|gaussian}  (constant => rho=1 for all lags) | Temporal autocorrelation model for the clutter Doppler process rho_clt(Delta_tau) |
| `rho_clt_bandwidth` | float | `0.1` | - | (0, 0.5] | Normalised clutter Doppler bandwidth (fraction of PRF); used by jakes/gaussian models |
| `clutter_as_deg` | float | `15.0` | deg | > 0 | Angular spread of clutter returns; controls spatial correlation C_a of clutter |
| `clutter_center_strategy` | str | `target_centroid` | - | {target_centroid|offset|uniform|random} | Spatial placement strategy for the clutter Power Angular Spectrum (PAS) at each transmit AP. Controls how clutter geometry couples to beam steering and therefore how kappa trades off clutter avoidance against beam quality. 'target_centroid': PAS centred on the AP->target-centroid direction (clutter and target spatially aligned; high kappa pushes the beam away from the target, collapsing both SCNR and SINR for any UE the target shadows). 'offset': PAS centred at (target azimuth + clutter_offset_az_deg), decoupling clutter avoidance from comm coverage. 'uniform': PAS spans the full azimuth (effectively C ~ (P_clt/M)*I); the clutter penalty becomes a global power regulariser and kappa no longer steers the beam. 'random': per-AP random azimuth drawn from U[-pi, pi]; useful for Monte-Carlo over clutter geometries. |
| `clutter_offset_az_deg` | float | `60.0` | deg | any real (typical: 30 to 90) | Azimuth offset added to the target azimuth when clutter_center_strategy='offset'; the clutter PAS is centred at (target azimuth + clutter_offset_az_deg). Larger values move the clutter further from the target direction, reducing the geometric coupling between sensing gain and clutter penalty. Ignored by other strategies. |

## Algorithm: CORDIS-ADMM

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `kappa` | float | `1.0` | - | >= 0 (0 = ignore clutter in objective) | Clutter penalty weight kappa in the linear sensing surrogate U_cpu^sens |
| `rho` | float | `1.0` | - | > 0 (typical: 0.1 to 10) | ADMM penalty parameter rho controlling consensus convergence speed |
| `n_max` | int | `50` | - | >= 1 | Maximum number of ADMM iterations N_max before forced termination |
| `eps_pri` | float | `0.001` | - | > 0 (typical: 1e-3 to 1e-5) | Primal residual convergence threshold epsilon_pri |
| `eps_dual` | float | `0.001` | - | > 0 (typical: 1e-3 to 1e-5) | Dual residual convergence threshold epsilon_dual |
| `xi_slack` | float | `10000.0` | - | >> 1 (typical: 1e3 to 1e5) | Slack variable penalty xi >> 0 in P-Central; ensures feasibility of SOC constraint |
| `target_priority_equal` | bool | `true` | - | {true|false} | Use equal priority weights omega_t = 1 for all targets; if false supply weights at runtime |
| `warm_start_from_split` | bool | `true` | - | {true|false} | Initialise ADMM beamformers W^(0) from CORDIS-Split Phase I output |
| `solver` | str | `CLARABEL` | - | {CLARABEL|GUROBI|MOSEK|SCS} | CVXPY backend solver for local QCQP subproblems (P-Local) |

## Algorithm: CORDIS-Split

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `epsilon_reg` | float | `0.01` | - | > 0 (typical: 1e-3 to 1e-1) | Regularisation parameter epsilon in LR-MMSE precoder (eq. split-comm-bf) |
| `epsilon_nsc` | float | `0.001` | - | > 0 (typical: 1e-4 to 1e-2) | Loading factor epsilon for null-space projection P_perp in NS-C beamformer |
| `target_priority_equal` | bool | `true` | - | {true|false} | Use equal priority weights omega_t = 1 for all targets in NS-C allocation |
| `kappa` | float | `1.0` | - | >= 0 | Clutter penalty weight kappa in the P-Split sensing utility U_cpu^sens |
| `xi_penalty` | float | `10000.0` | - | >> 1 (typical: 1e3 to 1e5) | Slack penalty xi >> 0 in P-Split; penalises violation of SINR QoS constraint |
| `solver` | str | `CLARABEL` | - | {CLARABEL|GUROBI|MOSEK|SCS} | CVXPY backend solver for P-Split convex program |

## Simulation Control

| Parameter | Type | Default | Unit | Range | Description |
|-----------|------|---------|------|-------|-------------|
| `n_trials` | int | `500` | - | >= 1 | Number of independent Monte Carlo realisations |
| `seed` | int | `42` | - | >= 0 | Master random seed; fully determines all random draws when fixed |
| `n_jobs` | int | `-1` | - | -1 or >= 1 | Number of parallel workers for Monte Carlo (-1 = all CPU cores) |
| `save_dir` | str | `results/` | - | any valid path | Root directory for saved result files (.npz + _meta.json pairs) |
| `tag` | str | `` | - | any string | Short label appended to output filenames for easy identification |
| `metrics` | List[str] | `['sinr_cdf', 'scnr_cdf', 'fronthaul', 'convergence']` | - | - |  |
| `benchmarks` | List[str] | `['centralized', 'mrt_centpa', 'zf_centpa', 'random_bf']` | - | - |  |
| `log_level` | str | `INFO` | - | {DEBUG|INFO|WARNING|ERROR} | Python logging level for the cordis.* logger hierarchy |
| `log_file` | Optional[str] | `null` | - | - |  |
| `progress_bar` | bool | `true` | - | - |  |
