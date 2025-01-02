from opencompass.cli.main import main
import pynvml 
from halo import Halo
import os 



def get_all_free_gpus():
    """
    获取当前所有空闲的 GPU 索引（显存占用小于 1024 MiB）。
    Returns:
        list: 所有空闲 GPU 的索引列表。
    """
    spinner = Halo(text='正在检查所有可用GPU...', spinner='dots')
    spinner.start()
    pynvml.nvmlInit()

    def is_gpu_free(handle):
        """判断 GPU 是否空闲（显存占用 < 1024 MiB）。"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory = mem_info.used / 1024 / 1024  # 将字节转换为 MiB
        return used_memory < 1024

    free_gpus = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()  # 获取总 GPU 数量
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            if is_gpu_free(handle):
                free_gpus.append(i)
        spinner.succeed(f"Found free GPUs: {free_gpus}")
    except pynvml.NVMLError as e:
        spinner.fail(f"NVML Error: {str(e)}")
    finally:
        pynvml.nvmlShutdown()

    return free_gpus


if __name__ == '__main__':
    free_gpus=get_all_free_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    main()
