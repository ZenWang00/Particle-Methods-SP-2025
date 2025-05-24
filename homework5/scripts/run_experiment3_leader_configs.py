import os
import sys
import numpy as np
import datetime
import traceback

# Add the parent directory (homework5) to sys.path to find the core simulation module
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
homework5_dir = os.path.dirname(scripts_dir) 
sys.path.insert(0, homework5_dir)

from zitianwang_LeaderFollowerParticleSystem import (
    run_leader_follower_simulation,
    # Default Physical/Sim Params
    L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT, 
    ETA_DEFAULT, DT_DEFAULT, T_COMMUNICATION_DEFAULT,
    # Default Convergence Params for LF
    POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
    AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT
)

log_header_written_exp3 = False
master_log_file_path_exp3 = ""

def write_to_master_log_exp3(log_data):
    global log_header_written_exp3, master_log_file_path_exp3
    if not master_log_file_path_exp3: return
    try:
        with open(master_log_file_path_exp3, "a") as f:
            if not log_header_written_exp3:
                header = ("Timestamp\tRunID\tType\tN_Followers\tN_Leaders\tCapacity\tT_Comm\tMax_Steps\tStartTime\tEndTime\tDuration_s\tStatus\tErrorMessage\n")
                f.write(header)
                log_header_written_exp3 = True
            duration_val = log_data.get('duration_s', 'N/A'); duration_str = f"{duration_val:.2f}" if isinstance(duration_val, (int,float)) else str(duration_val)
            f.write(f"{datetime.datetime.now().isoformat()}\t"
                    f"{log_data.get('run_id', 'N/A')}\tLeaderFollower\t"
                    f"{log_data.get('N_f', 'N/A')}\t{log_data.get('N_l', 'N/A')}\t"
                    f"{log_data.get('capacity', 'N/A')}\t{log_data.get('T_comm', 'N/A')}\t"
                    f"{log_data.get('max_steps', 'N/A')}\t"
                    f"{log_data.get('start_time','N/A').isoformat() if isinstance(log_data.get('start_time'),datetime.datetime) else 'N/A'}\t"
                    f"{log_data.get('end_time','N/A').isoformat() if isinstance(log_data.get('end_time'),datetime.datetime) else 'N/A'}\t"
                    f"{duration_str}\t{log_data.get('status','N/A')}\t"
                    f"{log_data.get('error_message','N/A').replace('\n',' ').replace('\r',' ')}\n")
    except Exception as e_log: print(f"Critical error writing to exp3 master log: {e_log}")

if __name__ == "__main__":
    main_output_folder_name = "simulation_outputs_exp3_leader_configs"
    main_output_folder_path = os.path.join(homework5_dir, main_output_folder_name)
    os.makedirs(main_output_folder_path, exist_ok=True)
    master_log_file_path_exp3 = os.path.join(main_output_folder_path, "_experiment3_master_log.tsv")
    print(f"All outputs for Experiment 3 will be saved under: {main_output_folder_path}/")

    # --- Experiment 3: Single vs. Multiple Leaders (Fixed Total Coverage) ---
    print("\n=== Running Experiment 3: Single vs. Multiple Leaders ===")
    
    # Fixed parameters for this experiment series
    N_f_exp3 = 500
    total_coverage_target_count = 150 # Target: 30% of 500 followers
    max_steps_exp3 = 5000 # Max steps for these runs, updated

    exp3_configs = [
        {"id": "Single-Leader",  "N_l": 1,  "capacity": total_coverage_target_count // 1, "max_steps": max_steps_exp3},
        {"id": "Multi-Leader-A", "N_l": 10, "capacity": total_coverage_target_count // 10, "max_steps": max_steps_exp3},
        {"id": "Multi-Leader-B", "N_l": 30, "capacity": total_coverage_target_count // 30, "max_steps": max_steps_exp3},
    ]

    for exp_config in exp3_configs:
        N_l_current = exp_config["N_l"]
        leader_cap_current = exp_config["capacity"]
        # Ensure total capacity matches the target
        if N_l_current * leader_cap_current != total_coverage_target_count and N_l_current > 0:
            print(f"Warning: For {exp_config['id']}, N_l={N_l_current}, cap={leader_cap_current}, total_cap={N_l_current*leader_cap_current} != {total_coverage_target_count}. Adjusting cap.")
            # This simple adjustment might not be perfect if total_coverage_target_count is not divisible
            leader_cap_current = int(np.ceil(total_coverage_target_count / N_l_current))
            print(f"Adjusted capacity to: {leader_cap_current}")
        
        run_id_str = f"{exp_config['id']}_Nf{N_f_exp3}_Nl{N_l_current}_cap{leader_cap_current}"
        print(f"\n--- Running Sub-Experiment: {run_id_str} ---")
        print(f"  N_f={N_f_exp3}, N_l={N_l_current}, Capacity={leader_cap_current}, Total Capacity={N_l_current*leader_cap_current}")

        lf_output_subdir_name = f"lf_exp3_{run_id_str}"
        lf_output_dir = os.path.join(main_output_folder_path, lf_output_subdir_name)
        os.makedirs(lf_output_dir, exist_ok=True)

        lf_params_current = {
            "N_f_run": N_f_exp3, "N_l_run": N_l_current, "L_domain_run": L_DOMAIN_DEFAULT, 
            "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
            "capacity_run": leader_cap_current, "T_comm_run": T_COMMUNICATION_DEFAULT, 
            "max_steps_run": max_steps_exp3, 
            "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
            "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
            "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
            "leader_dist_mode_run": "random",
            "output_dir": lf_output_dir
        }

        log_data = { "run_id": lf_output_subdir_name, "type": "LeaderFollower-Exp3", "N_f": N_f_exp3, "N_l": N_l_current, "capacity": leader_cap_current, "T_comm": T_COMMUNICATION_DEFAULT, "max_steps": max_steps_exp3 }
        log_data["start_time"] = datetime.datetime.now()
        try:
            run_leader_follower_simulation(**lf_params_current)
            log_data["status"] = "SUCCESS"
        except Exception as e: log_data["status"] = "ERROR"; log_data["error_message"] = traceback.format_exc(); print(f"ERROR in LF run {log_data['run_id']}: {e}")
        finally: 
            log_data["end_time"] = datetime.datetime.now()
            log_data["duration_s"] = (log_data["end_time"] - log_data["start_time"]).total_seconds()
            write_to_master_log_exp3(log_data)

    print("\nExperiment 3 (Single vs. Multiple Leaders) processing finished.")
    print(f"All outputs for Experiment 3 are in: {main_output_folder_path}") 