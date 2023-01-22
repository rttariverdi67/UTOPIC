from sklearn.neighbors import NearestNeighbors
import math
import numpy as np



def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()




class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M
        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.
        Args:
            num (int): Number of points to resample to, i.e. M
        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            
            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)


        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.
        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):

        refperm = np.random.permutation(sample['points_ref'].shape[0])
        srcperm = np.random.permutation(sample['points_src'].shape[0])
        sample['points_ref'] = sample['points_ref'][refperm, :]
        sample['points_src'] = sample['points_src'][srcperm, :]
        
        perm_mat = np.zeros((sample['points_src'].shape[0], sample['points_ref'].shape[0]))
        srcpermsort = np.argsort(srcperm)
        refpermsort = np.argsort(refperm)
        for i, j in zip(srcpermsort, refpermsort):
            perm_mat[i, j] = 1
        sample['perm_mat'] = perm_mat
        sample['src_overlap_gt'] = np.ones((sample['points_src'].shape[0], 1))
        sample['ref_overlap_gt'] = np.ones((sample['points_ref'].shape[0], 1))

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True
        return sample
