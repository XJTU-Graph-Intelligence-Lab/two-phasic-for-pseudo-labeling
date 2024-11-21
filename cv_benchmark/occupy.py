import torch

def allocate_tensor_on_gpus(gpu_ids, tensor_size_gb=10):
    # 计算张量的大小（以字节为单位）
    tensor_size_bytes = tensor_size_gb * 1024**3  # 10GB
    # 计算每个 GPU 上的张量大小
    tensor_size_per_gpu = tensor_size_bytes

    for gpu_id in gpu_ids:
        # 设置设备
        device = torch.device(f'cuda:{gpu_id}')
        
        # 分配张量
        tensor = torch.empty(tensor_size_per_gpu, dtype=torch.float32, device=device)
        print(f'Allocated {tensor_size_per_gpu / (1024**2):.2f} MB on GPU {gpu_id}')

if __name__ == "__main__":
    while True:
        user_input = input("Enter GPU IDs (comma separated) or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        try:
            gpu_id_list = list(map(int, user_input.split(',')))
            allocate_tensor_on_gpus(gpu_id_list)
        except ValueError:
            print("Invalid input. Please enter a comma-separated list of GPU IDs.")
        except RuntimeError as e:
            print(f"Error: {e}")