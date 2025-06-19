import matplotlib.patches as patches
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from random import randint


import random
import torch
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import models_YaTC
from pathlib import Path
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from torch.utils.data import DataLoader



class Marker:
    def __init__(self, ax):
        self.ax = ax
        
        # This is from ChatGPT, I checked some of the values
        self.PROPERTIES = {
            # IP header (assuming no options)
            "ver_ihl": [(0, 1)],
            "tos": [(1, 2)],
            "len": [(2, 4)],
            "id": [(4, 6)],
            "flags_frag": [(6, 8)],
            "ttl": [(8, 9)],
            "proto": [(9, 10)],
            "chksum": [(10, 12)],
            "ip_addr": [(12, 16), (16, 20)],   # [src_ip, dst_ip]

            # TCP header
            "port": [(20, 22), (22, 24)],      # [src_port, dst_port]
            "seq": [(24, 28)],
            "ack": [(28, 32)],
            "offset_flags": [(32, 34)],
            "tcp_win": [(34, 36)],
            "tcp_chksum": [(36, 38)],
            "urg_ptr": [(38, 40)],
        }

    

    def _mark_mfr_range(self, ranges: List[Tuple[int, int]], color, label=None) -> None:
        for packet_id in range(5):
            base_row = packet_id * 8 
            for i, (start, end) in enumerate(ranges):
                row = base_row + start // 40
                start = start % 40
                height = 1
                width = end - start
                curr_label = label if label and packet_id == 0 and i == 0 else None
                rect = patches.Rectangle(
                    (start - 0.5, row - 0.5),  # (x, y): left-most column - 0.5, row - 0.5
                    width, height,                  # width = 5 columns (cols 5–9), height = 1 row
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                    label = curr_label
                )
                self.ax.add_patch(rect)

                if label:
                    self.ax.legend()



    def _mark_mfr_property(self, property: str, color):
        ranges = self.PROPERTIES.get(property, None)
        if ranges:
            self._mark_mfr_range(ranges, color, label=property)
        else:
            raise ValueError(f"Unknown property: {property}")
        
    def _mark_multi_properties_mfr(self, properties: List[str]):
        colors = plt.cm.get_cmap('tab20', len(properties))
        for i, property in enumerate(properties):
            color = colors(i)
            self._mark_mfr_property(property, color)

    def _mark_all_properties_mfr(self):
        self._mark_multi_properties_mfr(list(self.PROPERTIES.keys()))

    def mark(self, properties: Union[List[str], None] = None):
        """
        Main marking function. can get a list of str properties,
        or leave blank to mark all
        """
        if properties:
            self._mark_multi_properties_mfr(properties)
        else:
            self._mark_all_properties_mfr()


