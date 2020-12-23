from utils import provider
import numpy as np
import torch


class AugmentationTransformer(object):
    def __init__(self, normal_channel):
        self.normal_channel = normal_channel

    def __call__(self, data):
        batch_data = np.array([data.pos.cpu().detach().numpy()])
        seq_number = data.x[:, 0].cpu().detach().numpy()
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        data_after_augmentation, shuffled_indices = provider.shuffle_points(rotated_data)
        seq_number = seq_number[shuffled_indices]
        data.pos = torch.from_numpy(data_after_augmentation[0]).to(data.pos.device)
        data.x = torch.from_numpy(np.insert(data_after_augmentation[0], 0, seq_number, axis=1)).to(data.x.device)
        return data
