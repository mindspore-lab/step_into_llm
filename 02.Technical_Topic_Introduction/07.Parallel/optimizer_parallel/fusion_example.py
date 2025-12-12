"""Parallel Optimizer Fusion Example"""
from mindspore.communication import init
from mindspore import nn
import mindspore as ms
init()

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, enable_parallel_optimizer=True)

class DenseLayer(nn.Cell):
    """A base layer with two dense layer"""
    def __init__(self):
        super().__init__()
        self.input_mapping = nn.Dense(10, 10)
        self.output_mapping = nn.Dense(10, 10)
    def construct(self, x):
        x = self.input_mapping(x)
        return self.output_mapping(x)

class Net(nn.Cell):
    """An network with many dense layers"""
    def __init__(self):
        super().__init__()
        self.layer1 = DenseLayer()
        self.layer2 = DenseLayer()
        self.layer3 = DenseLayer()
        self.layer1.set_comm_fusion(0)
        self.layer2.set_comm_fusion(1)
        self.layer3.set_comm_fusion(2)
    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net = Net()
for item in net.trainable_params():
    print(f"The parameter {item.name}'s fusion id is {item.comm_fusion}")
