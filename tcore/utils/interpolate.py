import torch
import torch.nn as nn
from pykeops import set_verbose
from pykeops.torch import Vi, Vj

set_verbose(False)

class knn_up(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, v_coor, v_feats, p_coor):
        """
        Input:
            v_coor: vox points coords [M, C]
            v_feats: vox feats [M, D]
            p_coor: points coords [N, C]
        Return:
            interpolated_feats: point_feats [N, D], N>M
        """
        # import ipdb;ipdb.set_trace()
        N, _ = p_coor.shape
        dists, idx = kNN(v_coor, p_coor, self.k)
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=1,
                         keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = torch.sum(index_points(v_feats, idx) *
                                       weight.view(N, self.k, 1),
                                       dim=1)
        return interpolated_feats


class knn_mask_interp(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, s_coor, s_mask, d_coor):
        """
        Input:
            s_coor: source coords [M, C]
            s_mask: source mask [M, D]
            d_coor: dest coords [N, C]
        Return:
            voted_mask: interpolated mask [N, D], N>M
        """

        _, idx = kNN(s_coor, d_coor, self.k)
        # get most voted value from the k neighbors
        voted_mask = torch.mode(s_mask[idx], dim=1,
                                keepdim=True).values.squeeze(1)
        return voted_mask


def kNN(x_train, x_test, K):
    # Encoding as KeOps LazyTensors:
    D = x_train.shape[-1]
    X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
    X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array
    # Symbolic distance matrix:
    D_ij = ((X_i - X_j)**2).sum(-1)

    KNN_ind = D_ij.argKmin(K, dim=1)
    indices = KNN_ind(x_test, x_train)
    pts = x_train[indices]
    values = ((pts - x_test.unsqueeze(1))**2).sum(-1)

    return values, indices


def index_points(points, idx):
    """
    Input:
        points: input point features [N, C]
        idx: sample index data [M]
    Return:
        new_points:, indexed features [M, C]
    """
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    new_points = points[idx, :]
    return new_points
