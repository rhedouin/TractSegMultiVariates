
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import sys
import shutil
from subprocess import call

import numpy as np
import nibabel as nib
from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage import binary_dilation
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import length as sl_length
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree

from tractseg.data import dataset_specific_utils
from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

from dipy.io.streamline import load_tractogram, save_tractogram, load_vtk, save_vtk
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines

from dipy.io.stateful_tractogram import StatefulTractogram

# from whitematteranalysis.io import read_polydata, write_polydata

import matplotlib
matplotlib.use('Agg')  # Solves error with ssh and plotting
#https://www.quora.com/If-a-Python-program-already-has-numerous-matplotlib-plot-functions-what-is-the-quickest-way-to-convert-them-all-to-a-way-where-all-the-plots-can-be-produced-as-hard-images-with-minimal-modification-of-code
import matplotlib.pyplot as plt
# Might fix problems with matplotlib over ssh (failing after connection is open for longer)
#   http://stackoverflow.com/questions/2443702/problem-running-python-matplotlib-in-background-after-ending-ssh-session
plt.ioff()


def plot_bundles_with_metric(bundle_path, atlas_path, endings_path, bundle, metrics, plot_3D_type, output_path):
    # Settings
    NR_SEGMENTS = 100
    ANTI_INTERPOL_MULT = 1  # increase number of points to avoid interpolation to blur the colors
    algorithm = "distance_map"  # equal_dist | distance_map | cutting_plane

    # Tractometry skips first and last element. Therefore we only have 98 instead of 100 elements.
    # Here we duplicate the first and last element to get back to 100 elements
    metrics = list(metrics)
    metrics = np.array([metrics[0]] + metrics + [metrics[-1]])

    metrics_max = metrics.max()
    metrics_min = metrics.min()

    # If all values identical, then scale_to_range does not work. Manually rescale to 0 if 0 or 99 if 1.
    if metrics_max == metrics_min:
        metrics *= 99
    else:
        metrics = img_utils.scale_to_range(metrics, range=(0, 99))  # range needs to be same as segments in colormap

    # Load mask
    beginnings_img = nib.load(endings_path)
    beginnings = beginnings_img.get_fdata().astype(np.uint8)
    for i in range(1):
        beginnings = binary_dilation(beginnings)

    # # Load trackings
    print(bundle_path)
    print(atlas_path)
    new_tract = load_tractogram(bundle_path, atlas_path)
    streamlines = new_tract.streamlines
    # new_streamlines = streamlines
    # sys.exit()

    # for jdx, sl in enumerate(streamlines):
    #     # print(streamlines[0])
    #     streamlines[jdx][0,:] = -streamlines[jdx][0,:]
    #     streamlines[jdx][1,:] = -streamlines[jdx][1,:]

    # save_tractogram(streamlines, "Comparaison_with_working_stuff.tck")
    # sys.exit()

    # Reduce streamline count
    streamlines = streamlines[::2]

    # Reorder to make all streamlines have same start region
    streamlines = list(transform_streamlines(streamlines, np.linalg.inv(beginnings_img.affine)))  # convert to voxel space
    streamlines = fiber_utils.orient_to_same_start_region(streamlines, beginnings)
    streamlines = list(transform_streamlines(streamlines, beginnings_img.affine))  # convert back to mm space


    if algorithm == "distance_map" or algorithm == "equal_dist":
        streamlines = fiber_utils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
    elif algorithm == "cutting_plane":
        streamlines = fiber_utils.resample_to_same_distance(streamlines, max_nr_points=NR_SEGMENTS,
                                                            ANTI_INTERPOL_MULT=ANTI_INTERPOL_MULT)

    # Cut start and end by percentage
    # streamlines = FiberUtils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
    # remove = int((NR_SEGMENTS * ANTI_INTERPOL_MULT) * 0.15)  # remove X% in beginning and end
    # streamlines = np.array(streamlines)[:, remove:-remove, :]
    # streamlines = list(streamlines)

    if algorithm == "equal_dist":
        segment_idxs = []
        for i in range(len(streamlines)):
            segment_idxs.append(list(range(NR_SEGMENTS * ANTI_INTERPOL_MULT)))
        segment_idxs = np.array(segment_idxs)

    elif algorithm == "distance_map":
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines)
        centroids = Streamlines(clusters.centroids)
        _, segment_idxs = cKDTree(centroids.get_data(), 1, copy_data=True).query(streamlines, k=1)

    elif algorithm == "cutting_plane":
        streamlines_resamp = fiber_utils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines_resamp)
        centroid = Streamlines(clusters.centroids)[0]
        # index of the middle cluster
        middle_idx = int(NR_SEGMENTS / 2) * ANTI_INTERPOL_MULT
        middle_point = centroid[middle_idx]
        segment_idxs = fiber_utils.get_idxs_of_closest_points(streamlines, middle_point)
        # Align along the middle and assign indices
        segment_idxs_eqlen = []
        for idx, sl in enumerate(streamlines):
            sl_middle_pos = segment_idxs[idx]
            before_elems = sl_middle_pos
            after_elems = len(sl) - sl_middle_pos
            base_idx = 1000  # use higher index to avoid negative numbers for area below middle
            r = range((base_idx - before_elems), (base_idx + after_elems))
            segment_idxs_eqlen.append(r)
        segment_idxs = segment_idxs_eqlen

    new_tract_path = os.path.dirname(output_path) + "/" + os.path.basename(output_path).split(".")[0] + "_" + bundle.split(".")[0] + "_resampled.tck"
    output_vtk =  os.path.dirname(output_path) + "/" + os.path.basename(output_path).split(".")[0]  + "_" + bundle.split(".")[0] +  "_resampled_with_" +  plot_3D_type + ".vtk"

    print("Save resampled tract")
    new_tract.streamlines = streamlines
    save_tractogram(new_tract,  new_tract_path)
    # print(os.path.basename(output_path).split(".")[0])

    # Put the way to your tckconvert
    print("Convert to vtk : " + output_vtk)
    tckconvert = "/home/rhedouin/Software/mrtrix3/bin/tckconvert"
    tckconvertCommand = [tckconvert, new_tract_path, output_vtk, "-force"]
    call(tckconvertCommand)
    f = open(output_vtk, "a")

    n = 0
    for jdx, sl in enumerate(streamlines):
        for idx, p in enumerate(sl):
            n = n+1

    # f.write("# vtk DataFile Version 1.0\n")
    # f.write("Data values for Tracks\n")
    # f.write("ASCII\n")
    # f.write("DATASET POLYDATA\n")
    # f.write("POINTS " + str(n) + " float\n")

    # for jdx, sl in enumerate(streamlines):
    #     colors_sl = []
    #     for idx, p in enumerate(sl):
    #         f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n")

    f.write("POINT_DATA " + str(n) + "\n")
    f.write("SCALARS " + plot_3D_type + " float 1\n")
    f.write("LOOKUP_TABLE my_table\n")

    for jdx, sl in enumerate(streamlines):
        colors_sl = []
        for idx, p in enumerate(sl):
            if idx >= len(segment_idxs[jdx]):
                seg_idx = segment_idxs[jdx][idx - 1]
            else:
                seg_idx = segment_idxs[jdx][idx]

            m = metrics[int(seg_idx / ANTI_INTERPOL_MULT)]
            n = n+1
            f.write(str(m) + "\n")

    f.close()   

