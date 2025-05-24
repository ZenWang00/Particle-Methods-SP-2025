import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_polarization_comparison(vicsek_csv, lf_csv, output_filename="polarization_comparison.png"):
    """
    Plots the polarization curves from Vicsek and Leader-Follower simulations on the same graph.
    """
    try:
        df_vicsek = pd.read_csv(vicsek_csv)
        df_lf = pd.read_csv(lf_csv)
    except FileNotFoundError as e:
        print(f"Error: One or both CSV files not found. {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    plt.figure(figsize=(12, 7))

    # Vicsek Polarization
    if 'Step' in df_vicsek.columns and 'FollowerPolarization' in df_vicsek.columns:
        plt.plot(df_vicsek['Step'], df_vicsek['FollowerPolarization'], label='Vicsek Model Polarization', color='green', marker='.', linestyle='-')
    else:
        print(f"Warning: Columns 'Step' or 'FollowerPolarization' not found in {vicsek_csv}")

    # Leader-Follower Polarization
    if 'Step' in df_lf.columns and 'FollowerPolarization' in df_lf.columns:
        plt.plot(df_lf['Step'], df_lf['FollowerPolarization'], label='Leader-Follower Polarization', color='blue', marker='x', linestyle='--')
    else:
        print(f"Warning: Columns 'Step' or 'FollowerPolarization' not found in {lf_csv}")
        
    plt.xlabel("Simulation Step")
    plt.ylabel("Global Follower Polarization ($\Phi$)")
    plt.title("Comparison of Polarization: Vicsek vs. Leader-Follower")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(output_filename)
        print(f"Polarization comparison plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving polarization comparison plot: {e}")
    plt.close()


def plot_leader_follower_all_metrics(lf_csv, output_filename="leader_follower_all_metrics.png"):
    """
    Plots all four convergence metrics for the Leader-Follower simulation.
    """
    try:
        df_lf = pd.read_csv(lf_csv)
    except FileNotFoundError:
        print(f"Error: {lf_csv} not found.")
        return
    except Exception as e:
        print(f"Error reading {lf_csv}: {e}")
        return

    required_cols = ['Step', 'AssignmentStability', 'FollowerPolarization', 'LeaderLoadStdDev', 'LeaderDistVariance']
    if not all(col in df_lf.columns for col in required_cols):
        print(f"Warning: One or more required columns ({required_cols}) are missing in {lf_csv}. Cannot plot all metrics.")
        # Try to plot available ones if Step is present
        # For simplicity, we just return if not all are present.
        return

    fig, axs = plt.subplots(4, 1, figsize=(14, 22), sharex=True)

    # Assignment Stability
    axs[0].plot(df_lf['Step'], df_lf['AssignmentStability'], label='Assignment Stability (AS)', color='purple', marker='o', markersize=3, linestyle='-')
    axs[0].set_ylabel("AS Ratio")
    axs[0].set_ylim(0, 1.05)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].legend()
    axs[0].set_title("Assignment Stability (Followers to Leaders)", fontsize=10)

    # Follower Polarization
    axs[1].plot(df_lf['Step'], df_lf['FollowerPolarization'], label='Follower Polarization (POL)', color='blue', marker='x', markersize=3, linestyle='--')
    axs[1].set_ylabel("Polarization ($\Phi$)")
    axs[1].set_ylim(0, 1.05)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].legend()
    axs[1].set_title("Global Follower Polarization", fontsize=10)

    # Leader Load Standard Deviation
    axs[2].plot(df_lf['Step'], df_lf['LeaderLoadStdDev'], label='Leader Load StdDev (LB_std)', color='orange', marker='s', markersize=3, linestyle='-.')
    axs[2].set_ylabel("StdDev of Followers per Leader")
    axs[2].set_ylim(bottom=0) # Auto-scale for y might be better here
    axs[2].grid(True, linestyle=':', alpha=0.7)
    axs[2].legend()
    axs[2].set_title("Leader Load Balance (Standard Deviation)", fontsize=10)
    
    # Leader Distance Variance
    axs[3].plot(df_lf['Step'], df_lf['LeaderDistVariance'], label='Leader Distance Variance (LS_var)', color='brown', marker='d', markersize=3, linestyle=':')
    axs[3].set_ylabel("Variance of Leader Distances")
    axs[3].set_ylim(bottom=0) # Auto-scale for y
    axs[3].set_xlabel("Simulation Step")
    axs[3].grid(True, linestyle=':', alpha=0.7)
    axs[3].legend()
    axs[3].set_title("Variance of Pairwise Leader Distances", fontsize=10)

    fig.suptitle("Leader-Follower Model: Convergence Metrics Evolution (N_f=500, N_l=20, Cap=7)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to make space for suptitle
    
    try:
        plt.savefig(output_filename)
        print(f"Leader-Follower all metrics plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving Leader-Follower all metrics plot: {e}")
    plt.close()


if __name__ == "__main__":
    # Assuming the CSV files are in the same directory as this script
    vicsek_csv_file = "vicsek_metrics.csv"
    lf_csv_file = "leader_follower_metrics.csv"

    # Check if pandas is available
    try:
        import pandas as pd
    except ImportError:
        print("Pandas library not found. Please install pandas to run this script: pip install pandas")
        exit()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib library not found. Please install matplotlib to run this script: pip install matplotlib")
        exit()

    plot_polarization_comparison(vicsek_csv_file, lf_csv_file)
    plot_leader_follower_all_metrics(lf_csv_file)

    print("\nPlotting script finished. Check for .png files in the current directory.") 