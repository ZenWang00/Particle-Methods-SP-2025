import os
import sys
import numpy as np
import datetime 
import traceback
import shutil

# Add the parent directory (homework5) to sys.path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
homework5_dir = os.path.dirname(scripts_dir)
sys.path.insert(0, homework5_dir)

from zitianwang_LeaderFollowerParticleSystem import (
    run_leader_follower_simulation,
    L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT, 
    ETA_DEFAULT, DT_DEFAULT, 
    POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
    AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT
)

if __name__ == "__main__":
    # --- Experiment: Low Leader Coverage with Large N_followers ---
    N_f_exp = 990  # 490 (original demo) + 500
    N_l_exp = 10   # Kept the same as demo
    capacity_exp = 10 # Kept the same as demo
    T_comm_exp = 10    # Kept the same as demo
    max_steps_exp = 150 # Max steps for this run, reduced to 150

    main_output_folder_name = "simulation_outputs_low_coverage_large_Nf"
    main_output_folder_path = os.path.join(homework5_dir, main_output_folder_name)
    
    run_id_str = f"Nf{N_f_exp}_Nl{N_l_exp}_cap{capacity_exp}_Tcomm{T_comm_exp}"
    output_subdir_name = f"lf_exp_{run_id_str}"
    output_dir = os.path.join(main_output_folder_path, output_subdir_name)

    if os.path.exists(output_dir):
        print(f"Cleaning up old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs for Low Coverage Large Nf experiment will be saved under: {output_dir}/")

    master_log_file_path = os.path.join(main_output_folder_path, "_low_coverage_large_Nf_master_log.tsv")
    log_header_written = os.path.exists(master_log_file_path)

    def write_to_master_log(log_data):
        global log_header_written, master_log_file_path
        try:
            with open(master_log_file_path, "a") as f:
                if not log_header_written:
                    header = ("Timestamp\tRunID\tType\tN_Followers\tN_Leaders\tCapacity\tT_Comm\tMax_Steps\tStartTime\tEndTime\tDuration_s\tStatus\tErrorMessage\n")
                    f.write(header)
                    log_header_written = True
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
        except Exception as e_log: print(f"Critical error writing to master log: {e_log}")

    print(f"\n=== Running Low Coverage Large Nf Experiment ===")
    print(f"  N_f={N_f_exp}, N_l={N_l_exp}, Capacity={capacity_exp}, T_comm={T_comm_exp}, MaxSteps={max_steps_exp}")

    lf_params = {
        "N_f_run": N_f_exp, "N_l_run": N_l_exp, "L_domain_run": L_DOMAIN_DEFAULT, 
        "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
        "capacity_run": capacity_exp, "T_comm_run": T_comm_exp, 
        "max_steps_run": max_steps_exp, 
        "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
        "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
        "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
        "leader_dist_mode_run": "random",
        "output_dir": output_dir
    }

    log_data = { 
        "run_id": output_subdir_name, "type": "LeaderFollower-LowCovLargeNf", 
        "N_f": N_f_exp, "N_l": N_l_exp, "capacity": capacity_exp, 
        "T_comm": T_comm_exp, "max_steps": max_steps_exp 
    }
    log_data["start_time"] = datetime.datetime.now()
    try:
        run_leader_follower_simulation(**lf_params)
        log_data["status"] = "SUCCESS"
    except Exception as e:
        log_data["status"] = "ERROR"
        log_data["error_message"] = traceback.format_exc()
        print(f"ERROR in Low Coverage Large Nf run {log_data['run_id']}: {e}")
    finally: 
        log_data["end_time"] = datetime.datetime.now()
        if "start_time" in log_data:
            log_data["duration_s"] = (log_data["end_time"] - log_data["start_time"]).total_seconds()
        write_to_master_log(log_data)

    print("\nLow Coverage Large Nf experiment finished.")
    print(f"Outputs are in: {output_dir}") 