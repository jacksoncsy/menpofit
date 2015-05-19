import menpo.io as mio
import numpy as np
from copy import deepcopy
from menpo.visualize import visualize_images, print_dynamic, progress_bar_str
from menpofit.visualize import plot_ced, visualize_fitting_results
from menpofit.regression.regressors import rls
import dill

src = r"/data/RA/incremental-alignment/CLM/data"

# tr_images_lfpw = mio.import_pickle(src + r"/mpie_trainset.pkl")
# tr_images_mpie = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\mpie_trainset.pkl")

# test_images_lfpw = mio.import_pickle(src + r"/lfpw_testset.pkl")
# init_shape_lfpw = mio.import_pickle(src + r"/init_shape_lfpw.pkl")

# test_images_helen = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\helen_testset.pkl")
# init_shape_helen = mio.import_pickle(r"C:\Csy\incremental-alignment\data\init_shape_helen.pkl")

from menpo.feature import no_op, sparse_hog, igo, lbp
# define my hog
def mySparseHog(img):
    return sparse_hog(img, cell_size=5, block_size=4)


from menpofit.sdm import SDMTrainer

# sdm_builder = SDMTrainer(n_levels=2, regression_features=mySparseHog, patch_shape=(20, 20),
#                          normalization_diagonal=250)
sdm_builder = SDMTrainer(n_levels=2, noise_std=0.04, regression_type=rls,
                         n_iterations=[3, 3], normalization_diagonal=200, n_perturbations=10)

# sdm_builder = SDMTrainer(n_levels=2,
#                          n_iterations=[1, 1], normalization_diagonal=200, n_perturbations=10)

# image_path = [src + r"/lfpw_trainset.pkl"]
# sdm = sdm_builder.train_csy([], image_path, verbose=True)

tr_images_lfpw = mio.import_pickle(src + r"/lfpw_trainset.pkl")
# lr_path = "/vol/hmi/projects/Shiyang/PAMI2015/code/"
sample_path = "/data/RA/incremental-alignment/SDM/models/"
# sdm, _ = sdm_builder.train_csy(tr_images_lfpw[:100], verbose=True, lr_path=sample_path, sample_path=sample_path)

trans_models = mio.import_pickle(sample_path + 'trans_models_ML.pkl')
sampling_stats = mio.import_pickle(sample_path + 'sdm_shog_lfpw/Sampling_STD_pdmAll_1.5.pkl')
# sdm = sdm_builder.train_parallel(tr_images_lfpw[:100], sampling_stats, trans_models, lr_save=True, verbose=True)

# lr_data_path = '/data/RA/incremental-alignment/SDM/models/sdm_parallel/sdm_parallel_shog_mpie_LR_data.pkl'
# lr_save_path = '/data/RA/incremental-alignment/SDM/models/sdm_iPar/tmp'
# sampling_factors = mio.import_pickle('/data/RA/incremental-alignment/SDM/models/sdm_iPar/sampling_factors/mpie2lfpw.pkl')
# sdm = sdm_builder.train_iPar(tr_images_lfpw[:100], sampling_stats, trans_models, lr_data_path,
#                              sampling_factors=sampling_factors, multi_step_rls=False,
#                              lr_save_path=lr_save_path, lr_save=True, verbose=True)


# sdm = sdm_builder.train(tr_images_lfpw[:400], verbose=True)
#
# with open('tmp_sdm_orig.pkl', 'wb') as f:
#    dill.dump(sdm, f)

with open('tmp_sdm_orig.pkl', 'rb') as f:
   sdm = dill.load(f)

# script function to perform fitting
def fitting(fitters,test_images,init_shapes,verbose=True):
    errs = [[] for i in range(len(fitters))]
    # fit images
    for fIdx in range(len(fitters)):
        for i in range(len(test_images)):
            gt = test_images[i]
            # obtain groubnd truth (original) landmarks
            gt_s = gt.landmarks['PTSX'].lms

            # fit image
            fr = fitters[fIdx].fit(gt, init_shapes[i], gt_shape=gt_s)

            # append fitting result to list
            errs[fIdx].append(fr.final_error())

            # print image numebr
            if verbose:
                print_dynamic('Fitter {}, Fitting - {} '.format(fIdx,
                    progress_bar_str((i + 1.)/len(test_images), show_bar=True)))
    return errs

i = tr_images_lfpw[0]
# obtain original landmarks
gt_s = i.landmarks['PTSX'].lms
# generate perturbed landmarks
s = sdm.perturb_shape(gt_s)
# fit image
fr = sdm.fit(i, s, gt_shape=gt_s)

print fr.final_error()