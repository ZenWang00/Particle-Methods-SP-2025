import os
import sys
import numpy as np
import datetime
import traceback

# Add the parent directory (homework5) to sys.path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
homework5_dir = os.path.dirname(scripts_dir)
sys.path.insert(0, homework5_dir)

from zitianwang_LeaderFollowerParticleSystem import (
    run_leader_follower_simulation,
    L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT, 
    ETA_DEFAULT, DT_DEFAULT, T_COMMUNICATION_DEFAULT,
    POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
    AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT
)

log_header_written_exp4 = False
master_log_file_path_exp4 = ""

def write_to_master_log_exp4(log_data):
    global log_header_written_exp4, master_log_file_path_exp4
    if not master_log_file_path_exp4: return
    try:
        with open(master_log_file_path_exp4, "a") as f:
            if not log_header_written_exp4:
                header = ("Timestamp\tRunID\tType\tN_Followers\tN_Leaders\tCapacity\tT_Comm\tMax_Steps\tLeaderDistMode\tStartTime\tEndTime\tDuration_s\tStatus\tErrorMessage\n")
                f.write(header)
                log_header_written_exp4 = True
            duration_val = log_data.get('duration_s', 'N/A'); duration_str = f"{duration_val:.2f}" if isinstance(duration_val, (int,float)) else str(duration_val)
            f.write(f"{datetime.datetime.now().isoformat()}\t"
                    f"{log_data.get('run_id', 'N/A')}\tLeaderFollower\t"
                    f"{log_data.get('N_f', 'N/A')}\t{log_data.get('N_l', 'N/A')}\t"
                    f"{log_data.get('capacity', 'N/A')}\t{log_data.get('T_comm', 'N/A')}\t"
                    f"{log_data.get('max_steps', 'N/A')}\t{log_data.get('leader_dist_mode','N/A')}\t"
                    f"{log_data.get('start_time','N/A').isoformat() if isinstance(log_data.get('start_time'),datetime.datetime) else 'N/A'}\t"
                    f"{log_data.get('end_time','N/A').isoformat() if isinstance(log_data.get('end_time'),datetime.datetime) else 'N/A'}\t"
                    f"{duration_str}\t{log_data.get('status','N/A')}\t"
                    f"{log_data.get('error_message','N/A').replace('\n',' ').replace('\r',' ')}\n")
    except Exception as e_log: print(f"Critical error writing to exp4 master log: {e_log}")

if __name__ == "__main__":
    main_output_folder_name = "simulation_outputs_exp4_initial_dist"
    main_output_folder_path = os.path.join(homework5_dir, main_output_folder_name)
    os.makedirs(main_output_folder_path, exist_ok=True)
    master_log_file_path_exp4 = os.path.join(main_output_folder_path, "_experiment4_master_log.tsv")
    print(f"All outputs for Experiment 4 will be saved under: {main_output_folder_path}/")

    print("\n=== Running Experiment 4: Impact of Initial Leader Distribution ===")
    
    # Fixed parameters for this experiment series
    N_f_exp4 = 500
    N_l_exp4 = 20
    capacity_exp4 = 8 # Approx 32% coverage (8*20=160 / 500)
    max_steps_exp4 = 5000 # Max steps for these runs, updated

    # Define different initial distribution modes to test
    distribution_modes = [
        {"id": "Dist-Random",   "mode": "random", "max_steps": max_steps_exp4},
        {"id": "Dist-Center",   "mode": "center_clustered", "max_steps": max_steps_exp4},
        {"id": "Dist-Grid",     "mode": "grid", "max_steps": max_steps_exp4},
        {"id": "Dist-Periphery","mode": "periphery", "max_steps": max_steps_exp4},
    ]

    for dist_config in distribution_modes:
        current_leader_dist_mode = dist_config["mode"]
        
        run_id_str = f"{dist_config['id']}_Nf{N_f_exp4}_Nl{N_l_exp4}_cap{capacity_exp4}"
        print(f"\n--- Running Sub-Experiment: {run_id_str} (Mode: {current_leader_dist_mode}) ---")

        lf_output_subdir_name = f"lf_exp4_{run_id_str}"
        lf_output_dir = os.path.join(main_output_folder_path, lf_output_subdir_name)
        os.makedirs(lf_output_dir, exist_ok=True)

        lf_params_current = {
            "N_f_run": N_f_exp4, "N_l_run": N_l_exp4, "L_domain_run": L_DOMAIN_DEFAULT, 
            "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
            "capacity_run": capacity_exp4, "T_comm_run": T_COMMUNICATION_DEFAULT, 
            "max_steps_run": max_steps_exp4, 
            "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
            "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
            "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
            "leader_dist_mode_run": current_leader_dist_mode, # Pass the distribution mode
            "output_dir": lf_output_dir
        }

        log_data = { "run_id": lf_output_subdir_name, "type": "LeaderFollower-Exp4", "N_f": N_f_exp4, "N_l": N_l_exp4, "capacity": capacity_exp4, "T_comm": T_COMMUNICATION_DEFAULT, "max_steps": max_steps_exp4, "leader_dist_mode": current_leader_dist_mode }
        log_data["start_time"] = datetime.datetime.now()
        try:
            run_leader_follower_simulation(**lf_params_current)
            log_data["status"] = "SUCCESS"
        except Exception as e: log_data["status"] = "ERROR"; log_data["error_message"] = traceback.format_exc(); print(f"ERROR in LF run {log_data['run_id']}: {e}")
        finally: 
            log_data["end_time"] = datetime.datetime.now()
            log_data["duration_s"] = (log_data["end_time"] - log_data["start_time"]).total_seconds()
            write_to_master_log_exp4(log_data)

    print("\nExperiment 4 (Initial Leader Distribution) processing finished.")
    print(f"All outputs for Experiment 4 are in: {main_output_folder_path}") 