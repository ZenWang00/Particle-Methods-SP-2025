import os
import sys
import numpy as np
import datetime 
import traceback
import shutil # For cleaning up old directories

# Add the parent directory (homework5) to sys.path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
homework5_dir = os.path.dirname(scripts_dir)
sys.path.insert(0, homework5_dir)

from zitianwang_LeaderFollowerParticleSystem import (
    run_pure_vicsek_simulation, # Added for Vicsek run
    run_leader_follower_simulation,
    # Default Physical/Sim Params that might be used if not overridden
    L_DOMAIN_DEFAULT, V0_DEFAULT, R_INTERACTION_DEFAULT, 
    ETA_DEFAULT, DT_DEFAULT, # T_COMMUNICATION_DEFAULT will be set by this script
    # Default Convergence Params for LF (can also be overridden if needed)
    POL_THRESH_LF_DEFAULT, CONV_WINDOW_LF_DEFAULT, CHECK_INTERVAL_LF_DEFAULT,
    AS_THRESH_DEFAULT, LB_THRESH_DEFAULT, LS_THRESH_DEFAULT,
    POL_THRESH_VICSEK_DEFAULT, CONV_WINDOW_VICSEK_DEFAULT, CHECK_INTERVAL_VICSEK_DEFAULT # Added Vicsek conv. params
)

if __name__ == "__main__":
    # --- General Demo Configuration ---
    TOTAL_PARTICLES_TARGET = 1000
    MAX_STEPS_DEMO_RUN = 300

    main_output_folder_name = "simulation_outputs_presentation_demo_N1000"
    main_output_folder_path = os.path.join(homework5_dir, main_output_folder_name)
    os.makedirs(main_output_folder_path, exist_ok=True)
    print(f"All outputs for this demo batch will be saved under: {main_output_folder_path}/")

    master_log_file_path = os.path.join(main_output_folder_path, "_demo_batch_master_log.tsv")
    log_header_written = os.path.exists(master_log_file_path) # Check if header needs to be written

    def write_to_master_log(log_data):
        global log_header_written, master_log_file_path
        # ... (Robust logging function as used in other experiment scripts)
        try:
            with open(master_log_file_path, "a") as f:
                if not log_header_written:
                    header = ("Timestamp\tRunID\tType\tN_Particles_Total\tN_Followers\tN_Leaders\tCapacity\tT_Comm\tMax_Steps\tStartTime\tEndTime\tDuration_s\tStatus\tErrorMessage\n")
                    f.write(header)
                    log_header_written = True
                duration_val = log_data.get('duration_s', 'N/A'); duration_str = f"{duration_val:.2f}" if isinstance(duration_val, (int,float)) else str(duration_val)
                f.write(f"{datetime.datetime.now().isoformat()}\t"
                        f"{log_data.get('run_id', 'N/A')}\t"
                        f"{log_data.get('type', 'N/A')}\t"
                        f"{log_data.get('N_total', 'N/A')}\t"
                        f"{log_data.get('N_f', 'N/A')}\t{log_data.get('N_l', 'N/A')}\t"
                        f"{log_data.get('capacity', 'N/A')}\t{log_data.get('T_comm', 'N/A')}\t"
                        f"{log_data.get('max_steps', 'N/A')}\t"
                        f"{log_data.get('start_time','N/A').isoformat() if isinstance(log_data.get('start_time'),datetime.datetime) else 'N/A'}\t"
                        f"{log_data.get('end_time','N/A').isoformat() if isinstance(log_data.get('end_time'),datetime.datetime) else 'N/A'}\t"
                        f"{duration_str}\t{log_data.get('status','N/A')}\t"
                        f"{log_data.get('error_message','N/A').replace('\n',' ').replace('\r',' ')}\n")
        except Exception as e_log: print(f"Critical error writing to master log: {e_log}")

    # --- 1. Pure Vicsek Simulation (N=1000) ---
    print("\n=== Running Pure Vicsek Simulation for Demo (N=1000) ===")
    N_vicsek_demo = TOTAL_PARTICLES_TARGET 
    vicsek_run_id = f"vicsek_N{N_vicsek_demo}"
    vicsek_output_dir = os.path.join(main_output_folder_path, vicsek_run_id)
    if os.path.exists(vicsek_output_dir): shutil.rmtree(vicsek_output_dir)
    os.makedirs(vicsek_output_dir, exist_ok=True)

    vicsek_params = {
        "N_particles": N_vicsek_demo, "L_domain_sim": L_DOMAIN_DEFAULT, "v0_sim": V0_DEFAULT,
        "R_interaction_sim": R_INTERACTION_DEFAULT, "eta_sim": ETA_DEFAULT, "dt_sim": DT_DEFAULT,
        "max_steps_sim": MAX_STEPS_DEMO_RUN, 
        "pol_thresh_sim": POL_THRESH_VICSEK_DEFAULT, "conv_window_sim": CONV_WINDOW_VICSEK_DEFAULT,
        "check_interval_sim": CHECK_INTERVAL_VICSEK_DEFAULT,
        "output_dir": vicsek_output_dir
    }
    log_data_vicsek = { "run_id": vicsek_run_id, "type": "PureVicsek", "N_total": N_vicsek_demo, "N_particles": N_vicsek_demo, "max_steps": MAX_STEPS_DEMO_RUN }
    log_data_vicsek["start_time"] = datetime.datetime.now()
    try:
        run_pure_vicsek_simulation(**vicsek_params)
        log_data_vicsek["status"] = "SUCCESS"
    except Exception as e: log_data_vicsek["status"] = "ERROR"; log_data_vicsek["error_message"] = traceback.format_exc(); print(f"ERROR in Vicsek Demo run: {e}")
    finally:
        log_data_vicsek["end_time"] = datetime.datetime.now()
        if "start_time" in log_data_vicsek: log_data_vicsek["duration_s"] = (log_data_vicsek["end_time"] - log_data_vicsek["start_time"]).total_seconds()
        write_to_master_log(log_data_vicsek)

    # --- 2. Leader-Follower Simulation (Low Coverage, Total N approx 1000) ---
    N_l_lf_demo = 10
    N_f_lf_demo = TOTAL_PARTICLES_TARGET - N_l_lf_demo # Should be 990
    capacity_lf_demo = 10 
    T_comm_lf_demo = 10   
    
    print(f"\n=== Running Leader-Follower Demo (Nf={N_f_lf_demo}, Nl={N_l_lf_demo}, Total={N_f_lf_demo+N_l_lf_demo}) ===")
    lf_run_id = f"lf_Nf{N_f_lf_demo}_Nl{N_l_lf_demo}_cap{capacity_lf_demo}_Tcomm{T_comm_lf_demo}"
    lf_output_dir = os.path.join(main_output_folder_path, lf_run_id)
    if os.path.exists(lf_output_dir): shutil.rmtree(lf_output_dir)
    os.makedirs(lf_output_dir, exist_ok=True)
    print(f"Outputs for LF Demo will be saved under: {lf_output_dir}/")

    lf_params = {
        "N_f_run": N_f_lf_demo, "N_l_run": N_l_lf_demo, "L_domain_run": L_DOMAIN_DEFAULT, 
        "v0_run": V0_DEFAULT, "R_interaction_run": R_INTERACTION_DEFAULT, "eta_run": ETA_DEFAULT, "dt_run": DT_DEFAULT,
        "capacity_run": capacity_lf_demo, "T_comm_run": T_comm_lf_demo, 
        "max_steps_run": MAX_STEPS_DEMO_RUN, 
        "pol_thresh_run": POL_THRESH_LF_DEFAULT, "conv_window_run_lf": CONV_WINDOW_LF_DEFAULT, 
        "check_interval_run_lf": CHECK_INTERVAL_LF_DEFAULT,
        "as_thresh_run": AS_THRESH_DEFAULT, "lb_thresh_run": LB_THRESH_DEFAULT, "ls_thresh_run": LS_THRESH_DEFAULT,
        "leader_dist_mode_run": "random",
        "output_dir": lf_output_dir
    }
    log_data_lf = { 
        "run_id": lf_run_id, "type": "LeaderFollower-Demo", "N_total": N_f_lf_demo + N_l_lf_demo,
        "N_f": N_f_lf_demo, "N_l": N_l_lf_demo, "capacity": capacity_lf_demo, 
        "T_comm": T_comm_lf_demo, "max_steps": MAX_STEPS_DEMO_RUN 
    }
    log_data_lf["start_time"] = datetime.datetime.now()
    try:
        run_leader_follower_simulation(**lf_params)
        log_data_lf["status"] = "SUCCESS"
    except Exception as e:
        log_data_lf["status"] = "ERROR"; log_data_lf["error_message"] = traceback.format_exc(); print(f"ERROR in LF Demo run: {e}")
    finally: 
        log_data_lf["end_time"] = datetime.datetime.now()
        if "start_time" in log_data_lf: log_data_lf["duration_s"] = (log_data_lf["end_time"] - log_data_lf["start_time"]).total_seconds()
        write_to_master_log(log_data_lf)

    print("\nAll Presentation Demo simulations finished.")
    print(f"Outputs are in: {main_output_folder_path}") 