
import torch
import numpy as np
from sklearn.cluster import KMeans


def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else np.array(x)

def proposals_to_centers(proposals):
    """
    :param proposals: [N, 5], (b_ix, x1, y1, x2, y2)
    :return: centers [N, 2], (b_ix, center_x, center_y)
    """
    cx = (proposals[:, 3] + proposals[:, 1]) / 2.0
    cy = (proposals[:, 4] + proposals[:, 2]) / 2.0
    center = np.vstack([cx, cy]).transpose()
    return center


def compute_cluster_targets(proposals, features, N_cluster=4, threshold=128):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...), N = 512
        features: [N, 4096],
    Return:
        batch_rois: [N_cluster, 128, 4096]
        batch_cluster_center: [N_cluster, 2], (center_x, center_y)
    '''

    proposals_np = to_np_array(proposals)
    features_np = to_np_array(features)
    centers = proposals_to_centers(proposals_np)

    """
    KMeans part
    """
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(centers)

    cluster_center = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    batch_rois_cluster = []
    for cluster_idx in range(0, N_cluster):
        keep_ix = np.where(cluster_labels[:] == cluster_idx)[0]

        if keep_ix.shape[0] < threshold:
            keep_ix_new = np.random.choice(keep_ix.shape[0], threshold, replace=True)
            keep_ix2 = keep_ix[keep_ix_new]
            batch_rois_tmp = features_np[keep_ix2]
        else:
            keep_ix2 = keep_ix[0:threshold]
            batch_rois_tmp = features_np[keep_ix2]


        # batch_rois_tmp = features[keep_ix]
        batch_rois_cluster.append(batch_rois_tmp)

    batch_rois_cluster = np.stack(batch_rois_cluster, axis=0) # (N_cluster, threshold, 4096)

    f = lambda x: (torch.from_numpy(x)).float().cuda(features.device).contiguous()
    batch_rois_cluster = f(batch_rois_cluster)
    # batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois_cluster, cluster_center

