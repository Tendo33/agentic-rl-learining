import json

import torch

from rllm.agents import Trajectory

# 加载 .pt 文件
data:list[Trajectory] = torch.load("search_trajectories.pt", weights_only=False)
# 打印全部内容

test_dat = data[213].to_dict()
json.dump(test_dat, open("test_data.json", "w"))


