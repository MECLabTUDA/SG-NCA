
import torch
import torch.nn as nn
from zeus import monitor
from models.LargeOctreeNCA2D import LargeOctreeNCA2D
from models.LargeNCA2D import LargeNCA2D
from models.surg_swin_unet import SurgicalSwinUNet
from models.surg_unet import SurgicalUNet
from models.surgical_segformer import SurgicalSegFormer
from training_utils.trainer import BaseTrainer, IncrementalNCATrainer, UNetTrainer
from zeus.monitor import ZeusMonitor
from torch.utils.flop_counter import FlopCounterMode
import matplotlib.pyplot as plt
from torch.utils.flop_counter import register_flop_formula

def large_nca_flops(module, args, kwargs, outputs):
    x = args[0]
    B, _, H, W = x.shape
    steps = module.base_nca.num_steps

    def conv_flops(conv):
        nnz = (conv.weight != 0).sum().item()
        return 2 * nnz * B * H * W

    flops_step = 0

    flops_step += conv_flops(module.base_nca.conv)
    flops_step += conv_flops(module.new_nca.conv)

    flops_step += conv_flops(module.base_nca.fc0)
    flops_step += conv_flops(module.new_nca.fc0)
    flops_step += conv_flops(module.base_nca.fc1)
    flops_step += conv_flops(module.new_nca.fc1)

    relu_ch = (
        module.base_nca.fc0.out_channels +
        module.new_nca.fc0.out_channels
    )
    flops_step += B * H * W * relu_ch

    return flops_step * steps

def get_energy_consumption(model: nn.Module, inp, train):
    model.train(train)
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

    monitor.begin_window("epoch")
    if train:
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(100):
            optimizer.zero_grad()
            model(inp).sum().backward()
            optimizer.step()
    else:
        with torch.no_grad():
            for _ in range(100):
                model(inp)
    torch.cuda.synchronize()
    mes = monitor.end_window("epoch")

    return {"energy": mes.gpu_energy[0]}

def get_flops(model: nn.Module, inp, train):
    model.train(train)
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(
        display=False,
        depth=None,
    )
    with flop_counter:
        if train:
            model(inp).sum().backward()
        else:
            with torch.no_grad():
                model(inp)
    total_flops = flop_counter.get_total_flops()
    return {"flops": total_flops}


trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dazzling-carnation-75") # NCA")
#trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/driven-oath-49")
#trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/effortless-water-55")
#trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/fragrant-frost-54")

#print(trainer.network)
#print(trainer.network.octree_nca.backbone_ncas[0].base_nca.fc1.weight[..., 0,0].cpu().numpy().shape)
#plt.imshow(trainer.network.octree_nca.backbone_ncas[0].base_nca.fc0.weight[..., 0,0].cpu().numpy() == 0)
#plt.show()
#plt.imshow(trainer.network.octree_nca.backbone_ncas[0].base_nca.fc1.weight[..., 0,0].cpu().numpy() == 0)
#plt.show()
#exit()

sample = torch.randn(1, 3, 256, 256).cuda() # (B, C, H, W)
pair_proposals = [torch.tensor([(1, 1) for _ in range(10)]).cuda()]

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
if isinstance(trainer.network, LargeOctreeNCA2D):
    trainer.network.merge_ncas()
    object_features_list = [torch.randn(5, 8, 165).cuda()] # List[Tensor(Num_Objs, T, Feat_Dim)]
elif isinstance(trainer.network, SurgicalSegFormer):
    object_features_list = [torch.randn(5, 8, 517).cuda()] # List[Tensor(Num_Objs, T, Feat_Dim)]
elif isinstance(trainer.network, SurgicalSwinUNet):
    object_features_list = [torch.randn(5, 8, 101).cuda()] # List[Tensor(Num_Objs, T, Feat_Dim)]
elif isinstance(trainer.network, SurgicalUNet):
    object_features_list = [torch.randn(5, 8, 552//8).cuda()] # List[Tensor(Num_Objs, T, Feat_Dim)]
else:
    raise NotImplementedError(f"Unknown network type for relation classifier input features: {trainer.network.__class__.__name__}")


my_monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

#torch.cuda.reset_peak_memory_stats()
#my_monitor.begin_window("epoch")
#start.record()
while True:
    trainer.network(sample) # segmentation
    trainer.relation_classifier(object_features_list, pair_proposals)


end.record()
mes = my_monitor.end_window("epoch")


torch.cuda.synchronize()
elapsed_time_ms = start.elapsed_time(end)
mem = torch.cuda.max_memory_allocated()
print(trainer.network.__class__.__name__)
print(f"Elapsed time: {elapsed_time_ms / 100:.2f} ms, FPS: {1000 / (elapsed_time_ms / 100):.2f}")
print(f"Peak memory usage: {mem / (1024 ** 2):.2f} MB")
print(f"Energy consumed: {mes.gpu_energy[0] / 100:.2f} mJ")

#print(trainer.network)
#print(trainer.network.octree_nca.backbone_ncas[0].base_nca.fc1.weight[..., 0,0].cpu().numpy().shape)