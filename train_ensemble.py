import subprocess
import sys

for i in range(1, 11):
    num = f"{i:02d}"
    cmd = ["python", "scripts/run.py", "--scene=./...", "--mode=nerf", "--train", "--n_steps=....", f"--save_training={num}", f"--save_snapshot=./../_{num}.msgpack"]
    ret_code = subprocess.run(cmd, check=True).returncode
    if ret_code != 0:
        print(f"Error running script for ensemble member {num}.")
        sys.exit(ret_code)

print("All ensemble members trained successfully.")
sys.exit(0)