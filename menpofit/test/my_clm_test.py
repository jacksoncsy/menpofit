import menpo.io as mio
import numpy as np
from copy import deepcopy
from menpo.visualize import visualize_images, print_dynamic, progress_bar_str
from menpofit.visualize import plot_ced, visualize_fitting_results

from menpofit.clm.classifier import linear_svm_lr, lda_lr, tk_lda_lr

# tr_images_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\lfpw_trainset.pkl")
# tr_images_mpie = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\mpie_trainset.pkl")

# tr_images_helen = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\helen_trainset.pkl")
#
# tr_images_MLH = deepcopy(tr_images_mpie)
# tr_images_MLH.extend(tr_images_lfpw)
# tr_images_MLH.extend(tr_images_helen)

test_images_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\lfpw_testset.pkl")
init_shape_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\data\init_shape_lfpw.pkl")

# test_images_helen = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\helen_testset.pkl")
# init_shape_helen = mio.import_pickle(r"C:\Csy\incremental-alignment\data\init_shape_helen.pkl")

from menpo.feature import no_op, sparse_hog, igo, lbp
# define my hog
def mySparseHog(img):
    return sparse_hog(img, cell_size=5, block_size=2)


# from menpofit.aam import AAMBuilder
# from menpo.feature import no_op, igo, hog
#
# # build model and use shape model only!
# shape_model_mpie = AAMBuilder(features=no_op,
#                  normalization_diagonal=100,
#                  n_levels=2).build(tr_images_mpie, verbose=True)


from menpofit.clm import CLMBuilder

# clm_builder = CLMBuilder(n_levels=1, features=no_op, patch_shape=(5, 5), patch_size=(3, 3))
# clm_builder = CLMBuilder(n_levels=2, features=mySparseHog, downscale=2, normalization_diagonal=200,
#                          patch_shape=(5, 5), classifier_trainers=tk_lda_lr)

# clm_builder = CLMBuilder(n_levels=2, features=no_op, downscale=2, normalization_diagonal=250,
#                          patch_shape=(5, 5), classifier_trainers=tk_lda_lr)

clm_builder = CLMBuilder(n_levels=1, features=mySparseHog, normalization_diagonal=100,
                         patch_shape=(5, 5), classifier_trainers=tk_lda_lr)

#
# clm = clm_builder.build(tr_images_lfpw, verbose=True)

# image_path = [r"C:\Csy\incremental-alignment\CLM\data\mpie_trainset.pkl"]
image_path = [r"C:\Csy\incremental-alignment\CLM\data\lfpw_trainset.pkl"]
              # r"C:\Csy\incremental-alignment\CLM\data\helen_trainset.pkl"]
clm = clm_builder.build([], image_path, verbose=True)
# mio.export_pickle(clm, r"C:\Csy\incremental-alignment\CLM\tmp\clm.pkl")

# clm = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\tmp\clm.pkl")

image_path = [r"C:\Csy\incremental-alignment\CLM\data\helen_trainset.pkl"]
# increment leadrning
clm.update_patch_expert([], image_path, verbose=True)


from menpofit.clm import GradientDescentCLMFitter

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

fitter1 = GradientDescentCLMFitter(clm, n_shape=[0.75])
err = fitting([fitter1], test_images_lfpw, init_shape_lfpw, verbose=True)