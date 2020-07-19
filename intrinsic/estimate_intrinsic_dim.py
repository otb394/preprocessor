from __future__ import print_function, division
import os, sys
import numpy as np
from os.path import join
from dataset import Config as DatasetConfig
from dataset import Dataset
import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Config(DatasetConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.est_config = "====== Intrinsic Dimension Estimation ======"
        self.log_dir = "exp/vanilla/log"
        # self.num_threads = 10
        self.log_filename = "est.log"
        # self.logrs = "-4:0:20"    # from -4 to 0 with 20 steps
        self.logrs = "-10:10:100"  # from -10 to 10 with 100 steps


class Solver(object):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.log_dir = config.log_dir
        # self.logrs = config.logrs
        
    def compute_pairwise_l2distance(self):
        print('ues L2 norm to compute the distance')
        num_samples = len(self.dataset)
        samples = self.dataset.samples
        # pdb.set_trace()
        R,C = np.triu_indices(num_samples,1)    # Denote N = num_samples
        pair_innerdot = np.einsum('ij,ij->i', samples[R,:], samples[C,:]) 
        # shape: (Nx(N-1)/2,) items are uptriangular part of mat [(Xi, Xj)], () denotes inner product
        norm = np.einsum('ij,ij->i', samples, samples)  # shape: (N,)
        return norm[R] + norm[C] - 2*pair_innerdot    
    
    def compute_pairwise_l1distance(self):
        print('ues L1 norm to compute the distance')
        num_samples = len(self.dataset)
        samples = self.dataset.samples
        R, C = np.triu_indices(num_samples, 1)  # Denote N = num_samples
        # R, C contain the row indexs and column indexs of upper-triangular part
        # shape: (Nx(N-1)/2,)
        l1norm = np.abs(samples[R, :] - samples[C, :]).sum(-1)

        return l1norm
        
    def compute_Cr(self, distances, r):
        return np.sum(distances < r) / len(distances)
        
    def show_curve(self, logrs, version = 1):
        start, end, step = logrs.split(":")
        assert int(step) > 0
        logrs = np.linspace(float(start), float(end), num=int(step))
        rs = np.exp(logrs)
        if (version == 1):
            distances = self.compute_pairwise_l1distance()
        else:
            distances = self.compute_pairwise_l2distance()

        logCrs = []
        for r in rs:
            logCrs.append(self.compute_Cr(distances, r))
        logCrs = np.log(np.array(logCrs))
        logCrs_d = (logCrs - logCrs[[*range(1,len(logCrs)), -1]]) / (logrs[0] - logrs[1])
        logCrs_d = logCrs_d[~np.isnan(logCrs_d)]
        logCrs_d = logCrs_d[np.isfinite(logCrs_d)]
        # remove the nan and inf from logCrs_d
        print("candidate estiamted instrinsic dim: {}".format(logCrs_d))
        ans = max(logCrs_d)
        print("the max instrinsic dim is: {}".format(max(logCrs_d)))
        return ans


def main(config):
    dataset = Dataset(config)
    print(">> creating solver")
    solver = Solver(dataset, config)
    print(">> solving")
    solver.show_curve(config.logrs)
    print(">> task finished")


if __name__ == '__main__':
    from utils import Logger
    config = Config()
    config.parse_args()
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
    config.print_args()
    main(config)

