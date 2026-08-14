# runs all the phases of the project in order, and checks for errors

import subprocess
import sys

def main():
    phases = [
        "phase1_single_particle.py",
        "phase2_two_particle_noninteracting.py",
        "phase3_interacting.py",
        "phase4_classical_comparison.py",
        "phase5_entropy_measure.py"
    ]

    for phase in phases:
        print("Running " + phase)
        subprocess.run([sys.executable, phase], check=True)

    print("All done! Check the plots folder")

if __name__ == "__main__":
    main()