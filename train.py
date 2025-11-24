import torch
import os
import time
import numpy as np


def main():
    print("=" * 40)
    print("🚀 STARTING TRAINING JOB")
    print("=" * 40)

    # 1. Check GPU
    print("\n[1] Checking Compute Resources...")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)  # Warmup
        print("   GPU Computation Test: SUCCESS")
    else:
        print("❌ NO GPU DETECTED! Training will be slow.")

    # 2. Check Data Access
    print("\n[2] Checking Data Access...")
    data_path = "/data"  # This is where we mounted the PVC
    if os.path.exists(data_path):
        print(f"✅ Data folder found at {data_path}")
        try:
            files = os.listdir(data_path)
            print(f"   Files found: {files[:5]}")

            # Create a dummy output file to prove write access
            with open(f"{data_path}/job_log_{int(time.time())}.txt", "w") as f:
                f.write("Hello from the cluster!")
            print("   Write Permission Test: SUCCESS")
        except Exception as e:
            print(f"⚠️  Access Warning: {e}")
    else:
        print(f"❌ Data folder not found at {data_path}")

    # 3. Simulate Training Loop
    print("\n[3] Simulating Training Loop...")
    for epoch in range(1, 6):
        time.sleep(1)  # Simulate work
        loss = np.random.random()
        print(f"   Epoch {epoch}/5 - Loss: {loss:.4f}")

    print("\n" + "=" * 40)
    print("🎉 JOB COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()