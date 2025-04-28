import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ising_results.txt", delimiter=",", skiprows=1)
T, M, E = data[:, 0], data[:, 1], data[:, 2]

# Plot Magnetization
plt.figure(figsize=(6,4))
plt.plot(T, M, marker="o", linestyle="-", color="b")
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization |M|")
plt.title("Ising Model: Magnetization vs. Temperature")
plt.grid(True)
plt.show()

# Plot Energy
plt.figure(figsize=(6,4))
plt.plot(T, E, marker="o", linestyle="-", color="r")
plt.xlabel("Temperature (T)")
plt.ylabel("Energy <E>")
plt.title("Ising Model: Energy vs. Temperature")
plt.grid(True)
plt.show()