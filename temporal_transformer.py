import math
import numpy as np
from scipy.spatial import distance as distance_calculator
import torch


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class TemporalTransformer(object):
    r"""Construct the edge index and add rotation invariant edge features (distance and angle).

    Args:
        k (int): The number of nearest neighbors to connect an edge between them
            from following frames.
            (default: :int:1)
        t (int): The number of following frames from which the algorithm selects
            the nearest neighbors.
            (default: :int:1)
    """

    def __init__(self, k=1, t=1):

        self.k = k
        self.t = t

    def __call__(self, data):
        edge_index = [[], []]
        edge_attributes = []
        for (source_index, point) in enumerate(data.x):
            mask = (data.x[:, 0] <= point[0] + self.t) & (data.x[:, 0] > point[0])
            original_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            points_in_following_frames = data.x[original_indices]
            if len(points_in_following_frames) <= 0:
                continue
            distances = distance_calculator.cdist(point[1:4].unsqueeze(0), points_in_following_frames[:, 1:4],
                                                  'euclidean')
            k_nearest_neighbor_indices = distances[0].argsort()[:self.k]
            k_nearest_neighbors = points_in_following_frames[k_nearest_neighbor_indices]
            k_nearest_neighbor_distances = distances[0][k_nearest_neighbor_indices]
            k_nearest_neighbor_angles = list(map(lambda nearest_point: angle_between(nearest_point[1:4], point[1:4]),
                                                 k_nearest_neighbors))
            k_nearest_neighbor_original_indices = original_indices[k_nearest_neighbor_indices]
            for index, distance, angle in zip(
                    *(k_nearest_neighbor_original_indices, k_nearest_neighbor_distances, k_nearest_neighbor_angles)):
                edge_index[0].append(source_index)
                edge_index[1].append(index)
                edge_attributes.append([distance, angle, math.sin(angle), math.cos(angle)])
            data.edge_index = torch.tensor(edge_index)
            data.edge_attr = torch.tensor(edge_attributes)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    print(math.sin(angle_between((1, 0, 0), (1, 0, 0))))
