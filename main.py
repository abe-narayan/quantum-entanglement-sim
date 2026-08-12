
# Runs all 5 phases of simulation


import subprocess

def main():
    phases = ["phase1.py", "phase2.py", "phase3.py", "phase4.py", "phase5.py"]
    for phase in phases:
        print("Running " + phase)
        subprocess.run([" python", phase])
    print("All done! Check the plots folder")

main()
