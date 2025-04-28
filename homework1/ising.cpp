#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

// Define constants for Ising Model
const int L = 10; // Lattice size
const int NTHERM = 100000;
const int NSAMPLE = 5000;
const int NSUBSWEEP = L * L;
const double TEMPERATURES[] = {0.0, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0};

void metropolis_ising() {
    vector<vector<int>> lattice(L, vector<int>(L, 1));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    ofstream output("ising_results.txt");
    output << "Temperature, Magnetization, Energy\n";

    for (double T : TEMPERATURES) {
        double magnetization = 0, energy = 0;
        
        // Thermalization steps
        for (int i = 0; i < NTHERM; ++i) {
            int x = rand() % L;
            int y = rand() % L;
            int deltaE = 2 * lattice[x][y] * (
                lattice[(x + 1) % L][y] + lattice[(x - 1 + L) % L][y] +
                lattice[x][(y + 1) % L] + lattice[x][(y - 1 + L) % L]
            );
            if (deltaE <= 0 || dis(gen) < exp(-deltaE / T)) {
                lattice[x][y] *= -1;
            }
        }
        
        // Sampling steps
        for (int i = 0; i < NSAMPLE; ++i) {
            for (int j = 0; j < NSUBSWEEP; ++j) {
                int x = rand() % L;
                int y = rand() % L;
                int deltaE = 2 * lattice[x][y] * (
                    lattice[(x + 1) % L][y] + lattice[(x - 1 + L) % L][y] +
                    lattice[x][(y + 1) % L] + lattice[x][(y - 1 + L) % L]
                );
                if (deltaE <= 0 || dis(gen) < exp(-deltaE / T)) {
                    lattice[x][y] *= -1;
                }
            }
            
            // Compute magnetization
            double sumM = 0;
            for (const auto& row : lattice) {
                for (int spin : row) {
                    sumM += spin;
                }
            }
            magnetization += abs(sumM) / (L * L);
            
            // Compute energy
            double sumE = 0;
            for (int x = 0; x < L; x++) {
                for (int y = 0; y < L; y++) {
                    sumE += -lattice[x][y] * (
                        lattice[(x + 1) % L][y] + lattice[(x - 1 + L) % L][y] +
                        lattice[x][(y + 1) % L] + lattice[x][(y - 1 + L) % L]
                    );
                }
            }
            energy += sumE / (L * L);
        }
        
        magnetization /= NSAMPLE;
        energy /= NSAMPLE;
        output << fixed << setprecision(4) << T << ", " << magnetization << ", " << energy << "\n";
    }
    output.close();
    cout << "Results saved to ising_results.txt" << endl;
}

int main() {
    metropolis_ising();
    return 0;
}

