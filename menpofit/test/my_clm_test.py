import menpo.io as mio
import numpy as np
from copy import deepcopy
from menpo.visualize import visualize_images, print_dynamic, progress_bar_str
from menpofit.visualize import plot_ced, visualize_fitting_results

tr_images_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\lfpw_trainset.pkl")
test_images_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\CLM\data\lfpw_testset.pkl")
init_shape_lfpw = mio.import_pickle(r"C:\Csy\incremental-alignment\data\init_shape_lfpw.pkl")

from menpo.feature import no_op, sparse_hog, igo, lbp
# define my hog
def mySparseHog(img):
    return sparse_hog(img, cell_size=5, block_size=2)

from menpofit.clm import CLMBuilder

clm_builder = CLMBuilder(n_levels=2,features=igo,patch_shape=(5,5))
clm = clm_builder.build(tr_images_lfpw, verbose=True)

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

fitter1 = GradientDescentCLMFitter(clm,n_shape=[0.75])
err = fitting([fitter1],test_images_lfpw,init_shape_lfpw,verbose=True)