# import torch
# import torch.nn as nn
# from transformers import RegNetModel
#
#
# from src.core import register
#
# __all__ = ['RegNet']
#
# @register
# class RegNet(nn.Module):
#     def __init__(self, configuration, return_idx=[0, 1, 2, 3]):
#         super(RegNet, self).__init__()
#         self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
#         self.return_idx = return_idx
#
#
#     def forward(self, x):
#
#         outputs = self.model(x, output_hidden_states = True)
#         x = outputs.hidden_states[2:5]
#
#         return x

import os
import torch
import torch.nn as nn
from transformers import RegNetModel, RegNetConfig
from src.core import register

__all__ = ['RegNet']


@register
class RegNet(nn.Module):
    def __init__(self, configuration=None, return_idx=[0, 1, 2, 3]):
        super(RegNet, self).__init__()

        # 尝试多种可能的本地路径
        possible_paths = [
            "./regnet-y-040-local",                                  # 当前目录
            "../regnet-y-040-local",                                 # 上级目录
            "../../regnet-y-040-local",                              # 上上级目录
            "c:/use/MAIN/STQS-DETR-main/regnet-y-040-local"            # 绝对路径
        ]
        
        local_path = None
        for path in possible_paths:
            if os.path.exists(path) and \
               os.path.exists(os.path.join(path, "config.json")) and \
               os.path.exists(os.path.join(path, "pytorch_model.bin")):
                local_path = path
                break

        self.return_idx = return_idx

        if local_path:
            print(f"Loading local RegNet weights from: {os.path.abspath(local_path)}")
            self.model = RegNetModel.from_pretrained(local_path)
        else:
            print("Warning: Local RegNet weights not found, trying official pre-trained model (facebook/regnet-y-040)")
            try:
                self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
            except Exception as e:
                print(f"Error: Could not load RegNet from web: {e}")
                print("Falling back to randomly initialized RegNet.")
                config = RegNetConfig.from_pretrained("facebook/regnet-y-040")
                self.model = RegNetModel(config)

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        x = outputs.hidden_states[2:5]
        return x
