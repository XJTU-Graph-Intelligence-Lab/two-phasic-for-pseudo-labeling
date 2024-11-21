import subprocess

commands = [
    "CUDA_VISIBLE_DEVICES=0 taskset -c 0-5 python3 optuna_run.py",
    "CUDA_VISIBLE_DEVICES=1 taskset -c 5-10 python3 optuna_run1.py",
    "CUDA_VISIBLE_DEVICES=2 taskset -c 10-15 python3 optuna_run2.py",
    "CUDA_VISIBLE_DEVICES=3 taskset -c 15-20 python3 optuna_run3.py",
    "CUDA_VISIBLE_DEVICES=0 taskset -c 20-25 python3 optuna_run4.py",
    "CUDA_VISIBLE_DEVICES=1 taskset -c 25-30 python3 optuna_run5.py",
]

# 为每个命令设置CUDA_VISIBLE_DEVICES并运行
processes = []
for cmd in commands:
    try:
        process = subprocess.Popen(
            f"{cmd}",
            shell=True
        )
        processes.append(process)
    except Exception as e:
        print(f"Error starting command '{cmd}': {e}")

# 等待所有进程完成
for process in processes:
    try:
        process.wait()
    except Exception as e:
        print(f"Error during execution: {e}")