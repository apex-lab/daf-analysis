from mne.stats.cluster_level import (
    _find_clusters,
    _cluster_indices_to_mask
)
from mne.parallel import parallel_func
from sklearn import neighbors
import numpy as np


class ClusterStatsOneTailed:
    '''
    Implements an upper/one-tailed cluster-based permutation test,
    or threshold-free cluster enhancement (TFCE) if a list of
    clustering thresholds is specified.

    This is an alternative API to some of MNE's internal
    functions, which allows us to use vertex-level test
    statistics from precomputed permutations (instead of
    MNE doing the permutations for us). This is handy,
    since we can use permutation schemes that MNE doesn't
    implement, such as shuffling blocks of TRs. And unlike
    nilearn's TFCE implementation, it works with arbitrary
    adjacency matrices, so we can perform clustering on
    the cortical surface (not just in volume space).
    '''

    def __init__(self, adjacency, threshold):
        '''
        Arguments
        ----------
        adjacency : scipy.sparse.spmatrix of shape (n_vertices, n_vertices)
            Specifies which vertices are next to one another for clustering.
        threshold : float or list[float]
            If float, `ClusterTest.perm_test` will implment a cluster
            based permutation test as in Maris & Oostenveld (2007) with
            specified threshold for cluster inclusion. If list of floats,
            will use MNE-like implmentation of threshold-free
            cluster enhancement (Smith & Nichols, 2009).
        '''
        self.adj = adjacency.tocoo()
        self.thres = threshold
        return None

    def _get_cluster_stats(self, x, threshold, t_power):
        clusters, cluster_stats = _find_clusters(
            x, threshold,
            tail = 1,
            adjacency = self.adj,
            max_step = 1,
            include = None,
            partitions = None,
            t_power = t_power,
            show_info = True
        )
        clusters  =_cluster_indices_to_mask(clusters, self.adj.shape[0])
        return np.array(clusters), cluster_stats

    def get_tfce_stats(self, x, E = .5, H = 2.):
        '''
        Computes TFCE stats for each vertex, given
        an (n_vertices,) observation `x`.

        Arguments
        ----------
        E : float, default: 0.5
            Exponential weight for extent. The canonical value is 0.5.
        H : float, default: 2.0
            Exponential weight for height. The canonical value is 2.0.

        Notes
        -------
        We use dh^H instead of h^H as in the original TFCE paper
        and in MNE. Other implementations (e.g. nilearn, FSLMaths)
        use h^H in the computation of the TFCE stat.
        This doesn't affect false positive rates, but it's
        something to be aware of -- there are actually two
        commonly used implementations of TFCE floating around!
        '''
        tfce_stats = np.zeros_like(x)
        assert(np.all(np.diff(self.thres) > 0)) # thresholds should be sorted
        last_thres = self.thres[0]
        for thres in self.thres[1:]:
            dh = thres - last_thres
            clusters, extent = self._get_cluster_stats(x, thres, 0)
            for c, e in zip(clusters, extent):
                tfce_stats[c] += (e**E) * (dh**H)
            last_thres = thres
        return tfce_stats

    def get_cluster_stats(self, x, t_power = 1):
        '''
        Arguments
        ------------
        x : np.array of shape (n_vertices,)
            Test statistic at each vertex/voxel.
        t_power : int, default: 1
            Exponent by which to raise test statistic at each vertex
            to before summing within a cluster. If 0,
            then cluster statistic is just count of vertices,
            if 1, then a sum, if 2, then squared sum, etc.
        '''
        if isinstance(self.thres, list):
            return None, self.get_tfce_stats(x)
        return self._get_cluster_stats(x, self.thres, t_power)

    def get_max_stat(self, x, t_power = 1):
        _, stats = self.get_cluster_stats(x, t_power)
        if stats.size == 0: # i.e. no clusters
            return 0.
        else:
            return np.max(stats)

    def perm_test(self, H0, t_power = 1, n_jobs = -1):
        '''
        Computes cluster statistics on precomputed permutation
        distributions of the test statistic for each voxel.
        Returned p-values are upper/one-tailed.

        Arguments
        ---------
        H0 : np.array of shape (n_permutations, n_vertices)
            The permutation distribution of the test statistic
            at each vertex/voxel. We assume that H0[0,:] is the
            observed test statistic.
        t_power : int, default: 1
            Exponent to raise test statistic at each vertex
            to before summing within a cluster. If 0,
            then cluster statistic is just a count, if
            1, then a sum, if 2, then squared sum, etc.
            This will be ignored if using TFCE.

        Returns
        ---------
        clusters : an (n_clusters, n_vertices) np.array or None
            Boolean masks indicating cluster membership.
            If using TFCE, this will be None.
        ps : an (n_clusters,) or (n_vertices,) np.array
            The p-values for each cluster for cluster-based permutation
            test or for each vertex for TFCE.
        H0_clust : np.array of shape (n_permutations,)
            The permutation null distribution of the
            maximum cluster statistic.
        '''
        parallel, p_func, n_jobs = parallel_func(
            self.get_max_stat,
            n_jobs = n_jobs,
            verbose = 1
        )
        out = parallel(
            p_func(H0[i, :], t_power)
            for i in range(H0.shape[0])
        )
        H0_clust = np.array(out)
        clusters, stats = self.get_cluster_stats(H0[0, :], t_power)
        clust_ps = np.array([(s <= H0_clust).mean() for s in stats])
        return clusters, clust_ps, H0_clust

def tfce(H0, adjacency, start = 0., step = .2):
    '''
    Arguments
    ---------
    H0 : np.array of shape (n_permutations, n_channels)
        The permutation distribution of the test statistic
        at each vertex/voxel. We assume that H0[0,:] is the
        observed test statistic.
    adjacency : scipy.sparse.spmatrix of shape (n_channels, n_channels)
        Specifies which channels are next to one another for clustering.

    Returns
    --------
    tfce_stat : float
        The maximum TFCE statistic on the observed data

    '''
    thresholds = np.arange(start, H0.max(), step).tolist()
    clust = ClusterStatsOneTailed(adjacency, thresholds)
    _, ps, H0_clust = clust.perm_test(H0)
    tfce_stat = H0_clust[0]
    return tfce_stat, ps
