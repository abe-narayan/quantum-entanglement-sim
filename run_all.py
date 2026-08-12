#runs all files in order
import subprocess

# List files wanting to run here
files = ["phase1.py", "phase2.py", "phase3.py", "phase4.py", "phase5.py"]

for name in files:
    print("running " + name + " ...")
    subprocess.run([" python", name])
    print(" Done with " + name)
    print()

print(" All phases finished! Check the plots folder.")
