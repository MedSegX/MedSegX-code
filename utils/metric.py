import torch.nn as nn

from monai.metrics import DiceMetric, SurfaceDiceMetric
from utils.HausdorffDistance import HausdorffDistanceMetric


class SegmentMetrics(nn.Module):
    def __init__(self, metric_name=["dsc"]):
        super().__init__()

        metric_dict = {}
        for name in metric_name:
            if name == "dsc":
                metric_dict[name] = DiceMetric()
            elif name == "nsd":
                metric_dict[name] = SurfaceDiceMetric(class_thresholds=[2])
            elif name == "hd":
                metric_dict[name] = HausdorffDistanceMetric(percentile=95.)
        self.metrics = metric_dict

    def forward(self, pred, target):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(pred, target).squeeze(1)
        return results