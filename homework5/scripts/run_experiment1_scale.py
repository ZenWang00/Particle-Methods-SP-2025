import os
import sys
import numpy as np
import datetime # For timestamping
import traceback # For detailed error logging
import shutil # Import shutil for rmtree

print("[DEBUG] Initial sys.path:", sys.path)
# Add project root to sys.path if not already there when running script from root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[DEBUG] Added project root {project_root} to sys.path")
print("[DEBUG] sys.path after potential modification:", sys.path)

# Ensure this script is run from the project root (e.g., Particle-Methods/)
# so that 'homework5' can be treated as a package.

try:
    from homework5.zitianwang_LeaderFollowerParticleSystem import (
        run_pure_vicsek_simulation,
        run_leader_follower_simulation,
        # Default Physical/Sim Params (if not overridden in configs below)
        L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT, 
        ETA_DEFAULT, DT_DEFAULT, T_COMMUNICATION_DEFAULT,
        # Default Convergence Params for LF
        POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
        AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT,
        # Default Convergence Params for Vicsek
        POL_THRESH_VICSEK_DEFAULT, CONV_WINDOW_VICSEK_DEFAULT, CHECK_INTERVAL_VICSEK_DEFAULT
    )
    print("[DEBUG] Successfully imported from homework5.zitianwang_LeaderFollowerParticleSystem")
except ModuleNotFoundError as e:
    print(f"[ERROR] Failed to import from homework5.zitianwang_LeaderFollowerParticleSystem. Error: {e}")
    print("    Troubleshooting steps:")
    print("    1. Ensure you are running this script from the PROJECT ROOT directory (e.g., Particle-Methods/).")
    print("       Your current working directory is: ", os.getcwd())
    print("    2. Ensure the 'homework5' directory is a direct sub-directory of your project root.")
    print("    3. Crucially, ensure the 'homework5' directory contains an empty file named '__init__.py'.")
    print(f"    4. The core module 'zitianwang_LeaderFollowerParticleSystem.py' exists directly inside 'homework5/'.")
    print(f"    Current sys.path for inspection: {sys.path}") # Print sys.path on error
    sys.exit(1)
except ImportError as e:
    print(f"[ERROR] General ImportError: {e}")
    sys.exit(1)

log_header_written_exp1 = False
master_log_file_path_exp1 = ""

def write_to_master_log_exp1(log_data):
    global log_header_written_exp1, master_log_file_path_exp1
    if not master_log_file_path_exp1: 
        print("Error: master_log_file_path_exp1 not set for Exp1.")
        return
    try:
        with open(master_log_file_path_exp1, "a") as f:
            if not log_header_written_exp1:
                header = ("Timestamp\tRunID\tType\tN_Particles\tN_Followers\tN_Leaders\t" +
                          "Capacity\tT_Comm\tMax_Steps\tStartTime\tEndTime\t" +
                          "Duration_s\tStatus\tErrorMessage\n")
                f.write(header)
                log_header_written_exp1 = True
            duration_val = log_data.get('duration_s', 'N/A'); duration_str = f"{duration_val:.2f}" if isinstance(duration_val, (int,float)) else str(duration_val)
            f.write(f"{datetime.datetime.now().isoformat()}\t"
                    f"{log_data.get('run_id', 'N/A')}\t"
                    f"{log_data.get('type', 'N/A')}\t"
                    f"{log_data.get('N_particles', 'N/A')}\t"
                    f"{log_data.get('N_f', 'N/A')}\t{log_data.get('N_l', 'N/A')}\t"
                    f"{log_data.get('capacity', 'N/A')}\t{log_data.get('T_comm', 'N/A')}\t"
                    f"{log_data.get('max_steps', 'N/A')}\t"
                    f"{log_data.get('start_time','N/A').isoformat() if isinstance(log_data.get('start_time'),datetime.datetime) else 'N/A'}\t"
                    f"{log_data.get('end_time','N/A').isoformat() if isinstance(log_data.get('end_time'),datetime.datetime) else 'N/A'}\t"
                    f"{duration_str}\t{log_data.get('status','N/A')}\t"
                    f"{log_data.get('error_message','N/A').replace('\n',' ').replace('\r',' ')}\n")
    except Exception as e_log: print(f"Critical error writing to exp1 master log: {e_log}")

if __name__ == "__main__":
    # Path for outputs should be relative to where this script is run from (project root)
    # So, outputs will go into Particle-Methods/homework5/simulation_outputs_exp1_scale/
    homework5_base_for_output = "homework5"
    main_output_folder_name = "simulation_outputs_exp1_scale"
    main_output_folder_path = os.path.join(homework5_base_for_output, main_output_folder_name) 
    os.makedirs(main_output_folder_path, exist_ok=True)
    print(f"All outputs for Experiment 1 will be saved under: {main_output_folder_path}/")

    master_log_file_path_exp1 = os.path.join(main_output_folder_path, "_experiment1_master_log.tsv")
    log_header_written_exp1 = os.path.exists(master_log_file_path_exp1)

    # --- Experiment 1: Impact of System Scale ---
    print("\n=== Running Experiment 1: Impact of System Scale ===")
    exp1_configs = [
        {"id": "Scale-S", "N_f": 80,   "N_l": 4,  "max_steps": 500, "target_coverage_ratio": 0.3},
        {"id": "Scale-M", "N_f": 800,  "N_l": 32, "max_steps": 1000, "target_coverage_ratio": 0.3},
        {"id": "Scale-L", "N_f": 8000, "N_l": 320,"max_steps": 1500, "target_coverage_ratio": 0.3}, 
    ]

    for exp_config in exp1_configs:
        N_f_current = exp_config["N_f"]
        N_l_current = exp_config["N_l"]
        max_steps_current = exp_config["max_steps"]
        target_coverage = exp_config["target_coverage_ratio"]
        if N_l_current > 0:
            num_to_cover = int(N_f_current * target_coverage); base_cap = num_to_cover // N_l_current; leader_cap_current = base_cap
            if num_to_cover > 0 and base_cap == 0: leader_cap_current = 1
            elif num_to_cover % N_l_current != 0 and base_cap*N_l_current < num_to_cover: leader_cap_current += 1
            if leader_cap_current == 0 and N_f_current > 0: leader_cap_current = 1
        else: leader_cap_current = float('inf')
        run_id_str = f"{exp_config['id']}_Nf{N_f_current}_Nl{N_l_current}"
        print(f"\n--- Running Sub-Experiment: {run_id_str} ---")

        # --- Vicsek Run with directory cleanup ---
        vicsek_output_subdir_name = f"vicsek_exp1_{run_id_str}"
        vicsek_output_dir = os.path.join(main_output_folder_path, vicsek_output_subdir_name)
        if os.path.exists(vicsek_output_dir):
            print(f"  Cleaning up old Vicsek output directory: {vicsek_output_dir}")
            shutil.rmtree(vicsek_output_dir)
        os.makedirs(vicsek_output_dir, exist_ok=True)
        
        vicsek_params_current = {
            "N_particles": N_f_current, "L_domain_sim": L_DOMAIN_DEFAULT, "v0_sim": V0_DEFAULT,
            "R_interaction_sim": R_INTERACTION_DEFAULT, "eta_sim": ETA_DEFAULT, "dt_sim": DT_DEFAULT,
            "max_steps_sim": max_steps_current, "pol_thresh_sim": POL_THRESH_VICSEK_DEFAULT, 
            "conv_window_sim": CONV_WINDOW_VICSEK_DEFAULT, "check_interval_sim": CHECK_INTERVAL_VICSEK_DEFAULT, 
            "output_dir": vicsek_output_dir
        }
        log_data_vicsek = { "run_id": vicsek_output_subdir_name, "type": "PureVicsek", "N_particles": N_f_current, "max_steps": max_steps_current, "N_f": N_f_current, "N_l": 0, "capacity": "N/A", "T_comm": "N/A" }
        log_data_vicsek["start_time"] = datetime.datetime.now()
        try:
            run_pure_vicsek_simulation(**vicsek_params_current)
            log_data_vicsek["status"] = "SUCCESS"
        except Exception as e: log_data_vicsek["status"] = "ERROR"; log_data_vicsek["error_message"] = traceback.format_exc(); print(f"ERROR in Vicsek run {log_data_vicsek['run_id']}: {e}")
        finally: 
            log_data_vicsek["end_time"] = datetime.datetime.now()
            if "start_time" in log_data_vicsek: log_data_vicsek["duration_s"] = (log_data_vicsek["end_time"] - log_data_vicsek["start_time"]).total_seconds()
            write_to_master_log_exp1(log_data_vicsek)

        # --- Leader-Follower Run with directory cleanup ---
        lf_output_subdir_name = f"lf_exp1_{run_id_str}_cap{leader_cap_current}"
        lf_output_dir = os.path.join(main_output_folder_path, lf_output_subdir_name)
        if os.path.exists(lf_output_dir):
            print(f"  Cleaning up old LF output directory: {lf_output_dir}")
            shutil.rmtree(lf_output_dir)
        os.makedirs(lf_output_dir, exist_ok=True)
        lf_params_current = {
            "N_f_run": N_f_current, "N_l_run": N_l_current, "L_domain_run": L_DOMAIN_DEFAULT, 
            "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
            "capacity_run": leader_cap_current, "T_comm_run": T_COMMUNICATION_DEFAULT, 
            "max_steps_run": max_steps_current, 
            "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
            "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
            "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
            "leader_dist_mode_run": "random",
            "output_dir": lf_output_dir
        }
        log_data_lf = { "run_id": lf_output_subdir_name, "type": "LeaderFollower-Exp1", "N_f": N_f_current, "N_l": N_l_current, "capacity": leader_cap_current, "T_comm": T_COMMUNICATION_DEFAULT, "max_steps": max_steps_current }
        log_data_lf["start_time"] = datetime.datetime.now()
        try:
            run_leader_follower_simulation(**lf_params_current)
            log_data_lf["status"] = "SUCCESS"
        except Exception as e: log_data_lf["status"] = "ERROR"; log_data_lf["error_message"] = traceback.format_exc(); print(f"ERROR in LF run {log_data_lf['run_id']}: {e}")
        finally: 
            log_data_lf["end_time"] = datetime.datetime.now()
            if "start_time" in log_data_lf: log_data_lf["duration_s"] = (log_data_lf["end_time"] - log_data_lf["start_time"]).total_seconds()
            write_to_master_log_exp1(log_data_lf)
            
    print("\nExperiment 1 (System Scale) processing finished.")
    print(f"All outputs for Experiment 1 are in: {main_output_folder_path}") 