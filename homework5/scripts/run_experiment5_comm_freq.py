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
    ETA_DEFAULT, DT_DEFAULT, # T_COMMUNICATION_DEFAULT will be varied
    POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
    AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT
)

log_header_written_exp5 = False
master_log_file_path_exp5 = ""

def write_to_master_log_exp5(log_data):
    global log_header_written_exp5, master_log_file_path_exp5
    if not master_log_file_path_exp5: return
    try:
        with open(master_log_file_path_exp5, "a") as f:
            if not log_header_written_exp5:
                header = ("Timestamp\tRunID\tType\tN_Followers\tN_Leaders\tCapacity\tT_Comm\tMax_Steps\tStartTime\tEndTime\tDuration_s\tStatus\tErrorMessage\n")
                f.write(header)
                log_header_written_exp5 = True
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
    except Exception as e_log: print(f"Critical error writing to exp5 master log: {e_log}")

if __name__ == "__main__":
    main_output_folder_name = "simulation_outputs_exp5_comm_freq"
    main_output_folder_path = os.path.join(homework5_dir, main_output_folder_name)
    os.makedirs(main_output_folder_path, exist_ok=True)
    master_log_file_path_exp5 = os.path.join(main_output_folder_path, "_experiment5_master_log.tsv")
    print(f"All outputs for Experiment 5 will be saved under: {main_output_folder_path}/")

    print("\n=== Running Experiment 5: Impact of Leader Communication Frequency ===")
    
    # Fixed parameters for this experiment series
    N_f_exp5 = 500
    N_l_exp5 = 20
    capacity_exp5 = 8 # Approx 32% coverage (8*20=160 / 500)
    max_steps_exp5 = 5000 # Max steps for these runs, updated

    # Define different T_communication values to test
    communication_intervals = [
        {"id": "Freq-VeryHigh", "T_comm": 5, "max_steps": max_steps_exp5},
        {"id": "Freq-Medium",   "T_comm": 30, "max_steps": max_steps_exp5},
        {"id": "Freq-None",     "T_comm": max_steps_exp5 + 1, "max_steps": max_steps_exp5}
    ]

    for comm_config in communication_intervals:
        current_T_comm = comm_config["T_comm"]
        
        run_id_str = f"{comm_config['id']}_Nf{N_f_exp5}_Nl{N_l_exp5}_cap{capacity_exp5}_Tcomm{current_T_comm if current_T_comm <= max_steps_exp5 else 'Inf'}"
        print(f"\n--- Running Sub-Experiment: {run_id_str} ---")

        lf_output_subdir_name = f"lf_exp5_{run_id_str}"
        lf_output_dir = os.path.join(main_output_folder_path, lf_output_subdir_name)
        os.makedirs(lf_output_dir, exist_ok=True)

        lf_params_current = {
            "N_f_run": N_f_exp5, "N_l_run": N_l_exp5, "L_domain_run": L_DOMAIN_DEFAULT, 
            "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
            "capacity_run": capacity_exp5, "T_comm_run": current_T_comm, 
            "max_steps_run": max_steps_exp5, 
            "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
            "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
            "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
            "leader_dist_mode_run": "random", # Using default random initial distribution for leaders
            "output_dir": lf_output_dir
        }
        
        log_data = { "run_id": lf_output_subdir_name, "type": "LeaderFollower-Exp5", "N_f": N_f_exp5, "N_l": N_l_exp5, "capacity": capacity_exp5, "T_comm": current_T_comm, "max_steps": max_steps_exp5 }
        log_data["start_time"] = datetime.datetime.now()
        try:
            run_leader_follower_simulation(**lf_params_current)
            log_data["status"] = "SUCCESS"
        except Exception as e: log_data["status"] = "ERROR"; log_data["error_message"] = traceback.format_exc(); print(f"ERROR in LF run {log_data['run_id']}: {e}")
        finally: 
            log_data["end_time"] = datetime.datetime.now()
            log_data["duration_s"] = (log_data["end_time"] - log_data["start_time"]).total_seconds()
            write_to_master_log_exp5(log_data)

    print("\nExperiment 5 (Leader Communication Frequency) processing finished.")
    print(f"All outputs for Experiment 5 are in: {main_output_folder_path}") 