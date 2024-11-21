import os
import json
import subprocess

commands = json.load(open('commands_new.json'))

# 创建输出目录
output_directory = "output_logs_optuna_ups"
os.makedirs(output_directory, exist_ok=True)

# 为每个命令设置CUDA_VISIBLE_DEVICES并运行
processes = []
for idx, cmd in enumerate(commands):
    output_file = os.path.join(output_directory, f"output_{idx}.log")  # 输出文件名
    with open(output_file, 'w') as outfile:
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=outfile,  # 重定向标准输出
                stderr=subprocess.STDOUT  # 将标准错误重定向到标准输出
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