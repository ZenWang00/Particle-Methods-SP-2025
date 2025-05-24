import pandas as pd
import matplotlib.pyplot as plt
import os

VISUAL_CONVERGENCE_THRESHOLD = 0.95 # Define the threshold for visual convergence

def plot_polarization_trends_comparison_highlighted(vicsek_csv_path, lf_csv_path, output_dir, output_filename="polarization_trends_highlighted_step100.png"):
    """
    Reads polarization data, plots trends, and highlights when polarization first exceeds a threshold.
    X-axis is limited to 100 steps.
    """
    try:
        df_vicsek = pd.read_csv(vicsek_csv_path)
        print(f"Read {len(df_vicsek)} rows from Vicsek CSV: {vicsek_csv_path}")
        df_lf = pd.read_csv(lf_csv_path)
        print(f"Read {len(df_lf)} rows from Leader-Follower CSV: {lf_csv_path}")

        plt.figure(figsize=(12, 8)) # Slightly taller figure for annotations

        # --- Plot Vicsek Model --- 
        first_vicsek_conv_step = None
        if not df_vicsek.empty and 'Step' in df_vicsek.columns and 'FollowerPolarization' in df_vicsek.columns:
            plt.plot(df_vicsek['Step'], df_vicsek['FollowerPolarization'], marker='.', linestyle='-', color='blue', label=f'Pure Vicsek (Threshold {VISUAL_CONVERGENCE_THRESHOLD})', markersize=4)
            vicsek_converged_rows = df_vicsek[df_vicsek['FollowerPolarization'] >= VISUAL_CONVERGENCE_THRESHOLD]
            if not vicsek_converged_rows.empty:
                first_vicsek_conv_step = vicsek_converged_rows['Step'].iloc[0]
                if first_vicsek_conv_step <= 100: # Only mark if within plot range
                    plt.axvline(x=first_vicsek_conv_step, color='blue', linestyle=':', linewidth=1)
                    plt.text(first_vicsek_conv_step + 1, VISUAL_CONVERGENCE_THRESHOLD - 0.05, f'{int(first_vicsek_conv_step)}', color='blue', ha='left')
        else:
            print(f"Warning: Vicsek data empty or missing columns in {vicsek_csv_path}")

        # --- Plot Leader-Follower Model --- 
        first_lf_conv_step = None
        if not df_lf.empty and 'Step' in df_lf.columns and 'FollowerPolarization' in df_lf.columns:
            plt.plot(df_lf['Step'], df_lf['FollowerPolarization'], marker='.', linestyle='--', color='red', label=f'Leader-Follower (Threshold {VISUAL_CONVERGENCE_THRESHOLD})', markersize=4)
            lf_converged_rows = df_lf[df_lf['FollowerPolarization'] >= VISUAL_CONVERGENCE_THRESHOLD]
            if not lf_converged_rows.empty:
                first_lf_conv_step = lf_converged_rows['Step'].iloc[0]
                if first_lf_conv_step <= 100: # Only mark if within plot range
                    plt.axvline(x=first_lf_conv_step, color='red', linestyle=':', linewidth=1)
                    plt.text(first_lf_conv_step + 1, VISUAL_CONVERGENCE_THRESHOLD - 0.1, f'{int(first_lf_conv_step)}', color='red', ha='left') # Adjusted y for LF label
        else:
            print(f"Warning: Leader-Follower data empty or missing columns in {lf_csv_path}")

        plt.xlabel("Simulation Step")
        plt.ylabel("Follower Polarization")
        title = f"Polarization Trends (Up to Step 100)\nHighlighting first step >= {VISUAL_CONVERGENCE_THRESHOLD} polarization"
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        plt.xlim(0, 100)
        
        os.makedirs(output_dir, exist_ok=True)
        full_output_path = os.path.join(output_dir, output_filename)
        plt.savefig(full_output_path)
        print(f"Highlighted polarization trend plot saved to: {full_output_path}")
        plt.close()

    except FileNotFoundError as e:
        print(f"Error: CSV file not found. {e}")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    vicsek_csv = os.path.join(current_script_dir, "simulation_outputs_presentation_demo_N1000", "vicsek_N1000", "vicsek_metrics.csv")
    lf_csv = os.path.join(current_script_dir, "simulation_outputs_presentation_demo_N1000", "lf_Nf990_Nl10_cap10_Tcomm10", "leader_follower_metrics.csv")
    plot_output_dir = os.path.join(current_script_dir, "simulation_outputs_presentation_demo_N1000")

    plot_polarization_trends_comparison_highlighted(vicsek_csv, lf_csv, plot_output_dir) 