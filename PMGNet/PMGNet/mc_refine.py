import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class ParamPredictor(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super(ParamPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出 alpha_c 和 log_sigma_c
        )

    def forward(self, mean, var):
        feat = torch.stack([mean, var], dim=1)  # (C,2)
        out = self.mlp(feat)  # (C,2)
        alpha = torch.sigmoid(out[:, 0])        # (0,1)
        sigma = torch.exp(out[:, 1])            # >0
        return alpha, sigma

def dynamic_threshold(prob_map, alpha):
    flat = prob_map.flatten()
    k = int(alpha * flat.numel())
    k = max(1, min(k, flat.numel() - 1))
    threshold = flat.kthvalue(k).values
    return threshold

def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device='cpu'):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2. * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_weighted_smoothing(prob_map, sigma=1.0):
    if len(prob_map.shape) == 3:
        prob_map = prob_map.unsqueeze(0).unsqueeze(0)
    device = prob_map.device
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)
    gaussian_kernel = gaussian_kernel_3d(kernel_size, sigma, device=device)
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    smoothed = F.conv3d(prob_map, gaussian_kernel, padding=kernel_size // 2)
    return smoothed.squeeze()

def refine_segmentation(prob_map, alpha, sigma):
    threshold = dynamic_threshold(prob_map, alpha)
    dynamic_mask = (prob_map >= threshold).float()
    smoothed_prob = gaussian_weighted_smoothing(prob_map, sigma=sigma)
    refined_prob = dynamic_mask * smoothed_prob + (1 - dynamic_mask) * prob_map
    return refined_prob

class TemperatureScaling(nn.Module):
    def __init__(self, num_channels, mutually_exclusive=False):
        super(TemperatureScaling, self).__init__()
        self.u = nn.Parameter(torch.zeros(num_channels))  # log(t_c)
        self.mutually_exclusive = mutually_exclusive

    def forward(self, prob_map):
        eps = 1e-6
        prob_map = torch.clamp(prob_map, eps, 1 - eps)
        logits = torch.log(prob_map / (1 - prob_map))  # logit

        # 计算通道级温度
        t = torch.exp(self.u).view(1, -1, *([1] * (logits.dim() - 2)))
        s = logits / t

        if self.mutually_exclusive:
            return F.softmax(s, dim=1)
        else:
            return torch.sigmoid(s)


class RefineSegmentationMultiChannel(nn.Module):
    def __init__(self, num_channels, mutually_exclusive=False):
        super(RefineSegmentationMultiChannel, self).__init__()
        self.param_predictor = ParamPredictor()
        self.temp_scaling = TemperatureScaling(num_channels, mutually_exclusive)
        self.num_channels = num_channels

    def forward(self, prob_map):
        """
        prob_map: (1,C,H,W,Z)
        return: (1,C,H,W,Z) refined probabilities (with temperature scaling)
        """
        assert prob_map.dim() == 5
        B, C = prob_map.shape[0], prob_map.shape[1]
        assert B == 1, "只支持 batch=1"

        refined_probs = []
        for c in range(C):
            prob_single = prob_map[0, c]  # (H,W,Z)
            mean = prob_single.mean()
            var = prob_single.var()

            alpha, sigma = self.param_predictor(mean.unsqueeze(0), var.unsqueeze(0))
            refined_prob = refine_segmentation(prob_single, alpha.item(), sigma.item())
            refined_probs.append(refined_prob)

        refined_probs = torch.stack(refined_probs, dim=0).unsqueeze(0)  # (1,C,H,W,Z)

        # 加温度缩放
        refined_probs = self.temp_scaling(refined_probs)

        return refined_probs
if __name__ == "__main__":

    x = torch.rand(1, 3, 16, 16, 8)

    model = RefineSegmentationMultiChannel(num_channels=3, mutually_exclusive=False)

    out = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("输出值范围: min={:.4f}, max={:.4f}".format(out.min().item(), out.max().item()))