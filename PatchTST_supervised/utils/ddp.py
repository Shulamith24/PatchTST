import torch.distributed as dist
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group


#检查分布式环境是否可用且已经初始化
def is_dist_avail_and_initialized():
    if not dist.is_available():     #如果分布式不可用(GPU不支持NCCL后端)
        return False
    if not dist.is_initialized():   #如果分布式进程组未初始化
        return False
    return True


def get_world_size():               #分布式训练中参与的进程数(GPU数量)
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



def get_rank():                     #获取当前进程的rank
    if not is_dist_avail_and_initialized(): #如果没初始化成功，当前进程就是主进程
        return 0
    return dist.get_rank()          #否则获取当前进程的rank


def is_main_process():              #判断当前进程是否是主进程
    return get_rank() == 0


def init_distributed_mode(args):    #初始化分布式训练模式，添加单卡模式选项
    if args.ddp:
        dist.init_process_group(backend="nccl")
        
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()          #获取当前进程rank，将当前进程绑定到一个GPU
        torch.cuda.set_device(local_rank)     
        torch.cuda.empty_cache()        ##清空GPU缓存
        print(f"Start running basic DDP on rank {rank}.")
        dist.barrier()                  #在所有进程同步位置
        #设置print限制
        setup_for_distributed(rank == 0)
    
    else:
        # 单卡模式不初始化分布式训练
        device_id = 0
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        print(f"使用GPU {device_id} 进行单卡调试。")
        return




def setup_for_distributed(is_master):   #设置分布式训练环境，确保只有主进程打印输出
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__          #获取python内置模块，其中包含了原始的print函数
    builtin_print = __builtin__.print       #获取原始print函数

    def print(*args, **kwargs):             #定义新的print函数为：只有在主进程时才打印
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def gather_tensors_from_all_gpus(tensor_list, device_id, to_numpy=True):
    """
    Gather tensors from all GPUs in a DDP setup onto each GPU.

    Args:
    local_tensors (list of torch.Tensor): List of tensors on the local GPU.

    Returns:
    list of torch.Tensor: List of all tensors gathered from all GPUs, available on each GPU.
    """
    world_size = dist.get_world_size()
    tensor_list = [tensor.to(device_id).contiguous() for tensor in tensor_list]
    gathered_tensors = [[] for _ in range(len(tensor_list))]

    # Gathering tensors from all GPUs
    for tensor in tensor_list:
        # Each GPU will gather tensors from all other GPUs
        gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, tensor)
        gathered_tensors.append(gathered_list)
    del tensor_list
    # Flattening the gathered list
    flattened_tensors = [
        tensor for sublist in gathered_tensors for tensor in sublist]
    del gathered_tensors
    if to_numpy:
        flattened_tensors_numpy = [tensor.cpu().numpy()
                                   for tensor in flattened_tensors]
        del flattened_tensors

        return flattened_tensors_numpy
    else:
        return flattened_tensors
