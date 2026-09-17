

from models.OctreeNCA2D import OctreeNCA2D


def get_cholec_model(num_classes, fire_rate) -> OctreeNCA2D:
    network = OctreeNCA2D(num_channels=20,
                      num_input_channels=3,
                      num_classes=num_classes,
                      hidden_size=64,
                      fire_rate=fire_rate,
                      num_steps=[10,10,10,10,20],
                      num_levels=5,
                      pool_op_kernel_sizes=[[2,2],[2,2],[2,2],[2,2]],
                      use_norm=False)
    network.do_ds = False
    return network