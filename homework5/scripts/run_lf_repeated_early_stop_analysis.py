import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import traceback
import shutil

# Add the parent directory (homework5) to sys.path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
homework5_dir = os.path.dirname(scripts_dir)
project_root_dir = os.path.dirname(homework5_dir) # Assuming homework5 is a direct child of project root for broader access if needed
sys.path.insert(0, homework5_dir)
sys.path.insert(0, project_root_dir) # Add project root for potential shared utilities if any

from zitianwang_LeaderFollowerParticleSystem import (
    run_leader_follower_simulation,
    # Import default physical parameters that will be used for the LF model
    L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT,
    ETA_DEFAULT, DT_DEFAULT, T_COMMUNICATION_DEFAULT, # Using default T_comm
    # Default LF convergence params are not strictly needed here as we define our own for early stopping
    # but can be a reference. We will define specific ones for this experiment.
)

# --- Experiment Configuration ---
NUM_REPEATS = 100  # Number of times to repeat the simulation
LEADER_DIST_MODE = "grid"  # Deterministic leader distribution: "grid", "center_clustered"
MAX_STEPS_PER_RUN = 300    # Maximum steps if early stopping criteria are not met

# Leader-Follower Model Parameters (align with run_presentation_demo.py LF part if desired, or set new)
# For this experiment, we will use the defaults from the main module,
# but some key ones related to particle numbers from run_presentation_demo.py for consistency.
N_L_RUN = 10
N_F_RUN = 990 # Total N = 1000
CAPACITY_RUN = 10
# T_COMM_RUN is T_COMMUNICATION_DEFAULT from import (30 steps)

# Early Stopping Convergence Criteria for LF Model in this experiment
LF_POL_THRESH_FOR_EARLY_STOP = 0.95
LF_CONV_WINDOW_FOR_EARLY_STOP = 5   # Shorter window for earlier stopping
LF_CHECK_INTERVAL_FOR_EARLY_STOP = 5 # Standard check interval

# Heatmap Parameters
BINS_HEATMAP = 50 # Number of bins for each dimension of the 2D histogram

# --- Main Execution ---
if __name__ == "__main__":
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"repeated_lf_early_stop_{LEADER_DIST_MODE}_Nf{N_F_RUN}_Nl{N_L_RUN}_cap{CAPACITY_RUN}_Tcomm{T_COMMUNICATION_DEFAULT}_reps{NUM_REPEATS}_{timestamp_str}"
    
    main_output_folder_path = os.path.join(homework5_dir, "simulation_outputs", experiment_name)
    if os.path.exists(main_output_folder_path):
        print(f"Warning: Output folder {main_output_folder_path} already exists. Consider cleaning or renaming.")
        # shutil.rmtree(main_output_folder_path) # Optional: Clean up old directory
    os.makedirs(main_output_folder_path, exist_ok=True)
    print(f"All outputs for this experiment batch will be saved under: {main_output_folder_path}")

    # --- Data Aggregation Lists ---
    aggregated_final_metrics_list = [] # List of final_metrics_dict from each run
    all_leader_positions_x = []        # All x-coords of all leaders across all steps from all runs
    all_leader_positions_y = []        # All y-coords of all leaders across all steps from all runs
    actual_stop_steps_list = []      # List of actual stopping steps for each run

    # --- Master Log File for this Batch ---
    master_log_file_path = os.path.join(main_output_folder_path, "_experiment_master_log.tsv")
    log_header_written = False

    def write_to_master_log(log_entry):
        global log_header_written
        try:
            with open(master_log_file_path, "a") as f:
                if not log_header_written:
                    headers = ["Timestamp", "RunID", "Status", "ActualSteps", 
                               "FinalPolarization", "FinalAssignStability", 
                               "FinalLeaderLoadStdDev", "FinalLeaderDistVar", "ErrorMessage"]
                    f.write("\t".join(headers) + "\n")
                    log_header_written = True
                
                row_data = [
                    datetime.datetime.now().isoformat(),
                    log_entry.get("run_id", "N/A"),
                    log_entry.get("status", "N/A"),
                    str(log_entry.get("actual_steps", "N/A")),
                    f"{log_entry.get('final_pol', 'N/A'):.4f}" if isinstance(log_entry.get('final_pol'), float) else "N/A",
                    f"{log_entry.get('final_as', 'N/A'):.4f}" if isinstance(log_entry.get('final_as'), float) else "N/A",
                    f"{log_entry.get('final_lb_std', 'N/A'):.4f}" if isinstance(log_entry.get('final_lb_std'), float) else "N/A",
                    f"{log_entry.get('final_ls_var', 'N/A'):.4f}" if isinstance(log_entry.get('final_ls_var'), float) else "N/A",
                    log_entry.get("error_message", "N/A").replace("\n", " ").replace("\r", " ")
                ]
                f.write("\t".join(row_data) + "\n")
        except Exception as e_log:
            print(f"Critical error writing to master log: {e_log}")

    print(f"\n=== Starting {NUM_REPEATS} Repeated Leader-Follower Simulations ===")
    for i_repeat in range(NUM_REPEATS):
        run_id = f"run_{i_repeat:03d}"
        print(f"--- Running: {run_id} ({i_repeat+1}/{NUM_REPEATS}) ---")
        
        current_run_output_dir = os.path.join(main_output_folder_path, run_id)
        os.makedirs(current_run_output_dir, exist_ok=True)

        log_entry = {"run_id": run_id}
        
        try:
            # Call the modified simulation function
            sim_results = run_leader_follower_simulation(
                N_f_run=N_F_RUN, N_l_run=N_L_RUN,
                L_domain_run=L_DOMAIN_DEFAULT,
                v0_run=V0_DEFAULT,
                R_interaction_run=R_INTERACTION_DEFAULT,
                eta_run=ETA_DEFAULT,
                dt_run=DT_DEFAULT,
                capacity_run=CAPACITY_RUN,
                T_comm_run=T_COMMUNICATION_DEFAULT,
                max_steps_run=MAX_STEPS_PER_RUN,
                pol_thresh_run=LF_POL_THRESH_FOR_EARLY_STOP, # Using early stop criteria
                conv_window_run_lf=LF_CONV_WINDOW_FOR_EARLY_STOP,
                check_interval_run_lf=LF_CHECK_INTERVAL_FOR_EARLY_STOP,
                as_thresh_run=0.95, # Default, not primary for stopping here
                lb_thresh_run=2.5,  # Default, not primary for stopping here
                ls_thresh_run=0.2,  # Default, not primary for stopping here
                output_dir=main_output_folder_path, # This is a fallback, override is used
                leader_dist_mode_run=LEADER_DIST_MODE,
                suppress_plots=True,         # Suppress internal plots for each run
                return_full_data=True,       # Get detailed data back
                output_dir_override=current_run_output_dir # Save this run's CSV to its own folder
            )

            if sim_results:
                final_metrics_dict, positions_history, leader_indices, types_arr, actual_steps = sim_results
                
                aggregated_final_metrics_list.append(final_metrics_dict)
                actual_stop_steps_list.append(actual_steps)
                
                # Extract leader positions from this run
                for step_positions in positions_history:
                    leader_pos_this_step = step_positions[leader_indices]
                    all_leader_positions_x.extend(leader_pos_this_step[:, 0])
                    all_leader_positions_y.extend(leader_pos_this_step[:, 1])
                
                log_entry["status"] = "SUCCESS"
                log_entry["actual_steps"] = actual_steps
                log_entry["final_pol"] = final_metrics_dict.get('follower_polarization')
                log_entry["final_as"] = final_metrics_dict.get('assignment_stability')
                log_entry["final_lb_std"] = final_metrics_dict.get('leader_load_std_dev')
                log_entry["final_ls_var"] = final_metrics_dict.get('leader_dist_variance')
            else:
                log_entry["status"] = "ERROR"
                log_entry["error_message"] = "Simulation function returned None/empty."

        except Exception as e:
            print(f"ERROR during {run_id}: {e}")
            traceback.print_exc()
            log_entry["status"] = "CRASH"
            log_entry["error_message"] = traceback.format_exc()
        finally:
            write_to_master_log(log_entry)

    print(f"\n=== All {NUM_REPEATS} Simulations Completed ===")

    # --- Post-processing and Analysis ---
    analysis_summary_path = os.path.join(main_output_folder_path, "_analysis_summary.txt")
    with open(analysis_summary_path, "w") as f_summary:
        print("\n--- Aggregated Metrics Analysis ---", file=f_summary)
        if aggregated_final_metrics_list:
            df_final_metrics = pd.DataFrame(aggregated_final_metrics_list)
            
            metrics_stats_csv_path = os.path.join(main_output_folder_path, "summary_metrics_statistics.csv")
            df_final_metrics.describe().to_csv(metrics_stats_csv_path)
            print(f"Final metrics descriptive statistics saved to: {metrics_stats_csv_path}", file=f_summary)
            print("\nDescriptive Statistics for Final Metrics:", file=f_summary)
            print(df_final_metrics.describe(), file=f_summary)
            
            # Print to console as well
            print("\nDescriptive Statistics for Final Metrics:")
            print(df_final_metrics.describe())
            print(f"Summary CSV saved to {metrics_stats_csv_path}")

        else:
            print("No final metrics data aggregated for statistics.", file=f_summary)
            print("No final metrics data aggregated for statistics.")

        print("\n--- Actual Stopping Steps Analysis ---", file=f_summary)
        if actual_stop_steps_list:
            avg_stop_step = np.mean(actual_stop_steps_list)
            std_stop_step = np.std(actual_stop_steps_list)
            min_stop_step = np.min(actual_stop_steps_list)
            max_stop_step = np.max(actual_stop_steps_list)
            print(f"Average stopping step: {avg_stop_step:.2f}", file=f_summary)
            print(f"Std Dev of stopping step: {std_stop_step:.2f}", file=f_summary)
            print(f"Min stopping step: {min_stop_step}", file=f_summary)
            print(f"Max stopping step: {max_stop_step}", file=f_summary)
            
            # Print to console
            print(f"\nAverage stopping step: {avg_stop_step:.2f}")
            print(f"Std Dev of stopping step: {std_stop_step:.2f}")

            plt.figure(figsize=(10, 6))
            plt.hist(actual_stop_steps_list, bins=max(10, NUM_REPEATS // 5), edgecolor='black')
            plt.title(f"Distribution of Actual Stopping Steps (N={NUM_REPEATS} runs)")
            plt.xlabel("Number of Simulation Steps at Stop")
            plt.ylabel("Frequency")
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            stop_steps_hist_path = os.path.join(main_output_folder_path, "stopping_steps_histogram.png")
            plt.savefig(stop_steps_hist_path)
            plt.close()
            print(f"Stopping steps histogram saved to: {stop_steps_hist_path}", file=f_summary)
            print(f"Stopping steps histogram saved to: {stop_steps_hist_path}")
        else:
            print("No stopping step data aggregated.", file=f_summary)
            print("No stopping step data aggregated.")

        print("\n--- Leader Position Heatmap Generation ---", file=f_summary)
        if all_leader_positions_x and all_leader_positions_y:
            plt.figure(figsize=(10, 8))
            # Create the 2D histogram
            counts, xedges, yedges, im = plt.hist2d(
                all_leader_positions_x, 
                all_leader_positions_y, 
                bins=BINS_HEATMAP, 
                cmap='viridis', # 'viridis', 'jet', 'hot' are good options
                range=[[0, L_DOMAIN_DEFAULT], [0, L_DOMAIN_DEFAULT]] # Ensure range matches domain
            )
            plt.colorbar(im, label='Leader Visit Frequency') # im is the Image object returned by hist2d
            plt.title(f"Leader Position Visit Frequency (All Steps, {NUM_REPEATS} Runs, Early Stopping)")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.gca().set_aspect('equal', adjustable='box') # Ensure square plot for square domain
            heatmap_path = os.path.join(main_output_folder_path, "leader_position_heatmap_early_stop.png")
            plt.savefig(heatmap_path)
            plt.close()
            print(f"Leader position heatmap saved to: {heatmap_path}", file=f_summary)
            print(f"Leader position heatmap saved to: {heatmap_path}")
        else:
            print("No leader position data aggregated for heatmap.", file=f_summary)
            print("No leader position data aggregated for heatmap.")

    print(f"\nExperiment finished. All summary outputs and plots are in: {main_output_folder_path}")
    print(f"Detailed run CSVs are in subfolders like: {os.path.join(main_output_folder_path, 'run_000')}") 