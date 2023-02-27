import argparse
import math
import os
import sys
from decimal import Decimal
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from hotelling.stats import hotelling_t2
from tqdm import tqdm
from tractseg.data import dataset_specific_utils
from tractseg.libs import metric_utils
from tractseg.libs import tracking

from AFQ_MultiCompCorrectionMultiVariates import get_significant_areas, AFQ_MultiCompCorrectionMultiVariates

import plot_utils_with_saving

# import statsmodels as sm

def parse_subjects_file(file_path):
    with open(file_path) as f:
        l = f.readline().strip()
        if l.startswith("# tractometry_path="):
            base_path = l.split("=")[1]
        else:
            raise ValueError("Invalid first line in subjects file. Must start with '# tractometry_path='")

        l = f.readline().strip()
        if l.startswith("# nb_variable="):
            nb_variable = int(l.split("=")[1])
            print("Consider " + str(nb_variable) + " variable")
        else:
            raise ValueError("Must set the number of variables")

        bundles = None
        plot_3D_path = None

        # parse bundle names
        for i in range(3):
            l = f.readline().strip()
            if l.startswith("# bundles="):
                bundles_string = l.split("=")[1]
                bundles = bundles_string.split(" ")

                valid_bundles = dataset_specific_utils.get_bundle_names("All_tractometry")[1:]
                for bundle in bundles:
                    if bundle not in valid_bundles:
                        raise ValueError("Invalid bundle name: {}".format(bundle))

                print("Using {} manually specified bundles.".format(len(bundles)))
            elif l.startswith("# plot_3D="):
                plot_3D_path = l.split("=")[1]

        if bundles is None:
            bundles = dataset_specific_utils.get_bundle_names("All_tractometry")[1:]

    df = pd.read_csv(file_path, sep=" ", comment="#")
    df["subject_id"] = df["subject_id"].astype(str)

    # Check that each column (except for first one) is correctly parsed as a number
    for col in df.columns[1:]:
        if not np.issubdtype(df[col].dtype, np.number):
            raise IOError("Column {} contains non-numeric values".format(col))

    if df.columns[1] == "group":
        if df["group"].max() > 1:
            raise IOError("Column 'group' may only contain 0 and 1.")

    print(bundles)
    return base_path, df, nb_variable, bundles, plot_3D_path


def correct_for_confounds(values, meta_data, nb_variable, bundles, selected_bun_indices, NR_POINTS, analysis_type,
                          confound_names):
    if len(confound_names) == 0:
        return values

    values_cor = np.zeros([len(bundles), NR_POINTS, nb_variable, len(meta_data)])

    for b_idx in selected_bun_indices:
        for var in range(nb_variable):
            for jdx in range(NR_POINTS):
                # only one feature -> have to add empty dimension because unconfound expects 2D array
                target = np.array([values[s][b_idx][jdx][var] for s in meta_data["subject_id"]])[
                    ..., None]  # [samples, 1]

                if analysis_type == "group":
                    target_cor = metric_utils.unconfound(target, meta_data[["group"] + confound_names].values,
                                                         group_data=True).squeeze()
                else:
                    target_cor = metric_utils.unconfound(target, meta_data[confound_names].values,
                                                         group_data=False).squeeze()
                    meta_data["target"] = metric_utils.unconfound(meta_data["target"].values[..., None],
                                                                  meta_data[confound_names].values,
                                                                  group_data=False).squeeze()
                values_cor[b_idx, jdx, var, :] = target_cor

    # Restore original data structure
    values_cor = values_cor.transpose(3, 0, 1, 2)
    # todo: nicer way: use numpy array right from beginning instead of dict
    values_cor_dict = {}
    for idx, subject in enumerate(list(meta_data["subject_id"])):
        values_cor_dict[subject] = values_cor[idx]

    return values_cor_dict


def get_corrected_alpha(values_allp: object, meta_data: object, analysis_type: object, subjects_A: object, subjects_B: object, alpha: object, bundles: object, nperm: object,
                        b_idx: object) -> object:
    if analysis_type == "group":
        y = np.array((0,) * len(subjects_A) + (1,) * len(subjects_B))
    else:
        y = meta_data["target"].values
    alphaFWE, statFWE, clusterFWE, stats = AFQ_MultiCompCorrectionMultiVariates(np.array(values_allp), y,
                                                                      alpha, nperm=nperm)
    return alphaFWE, clusterFWE


def format_number(num):
    if abs(num) > 0.00001:
        return round(num, 6)
    else:
        return '%.2e' % Decimal(num)


def plot_tractometry_with_pvalue(values, meta_data, nb_variable, bundles, selected_bundles, output_path, alpha,
                                 FWE_method,
                                 analysis_type, correct_mult_tract_comp, show_detailed_p, nperm=1000,
                                 hide_legend=False, plot_3D_path=None, plot_3D_type="none",
                                 tracking_format="trk_legacy", tracking_dir="auto", atlas_path="", show_color_bar=True,
                                 save_csv=False, y_range=None):
    NR_POINTS = values[meta_data["subject_id"][0]].shape[1]
    selected_bun_indices = [bundles.index(b) for b in selected_bundles]
    print(selected_bun_indices)

    if analysis_type == "group":
        subjects_A = list(meta_data[meta_data["group"] == 0]["subject_id"])
        subjects_B = list(meta_data[meta_data["group"] == 1]["subject_id"])
    else:
        subjects_A = list(meta_data["subject_id"])
        subjects_B = []

    confound_names = list(meta_data.columns[2:])

    cols = 5
    rows = math.ceil(len(selected_bundles) / cols)

    a4_dims = (cols * 3, rows * 5)
    f, axes = plt.subplots(rows, cols, figsize=a4_dims)

    axes = axes.flatten()
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Correct for confounds.
    values = correct_for_confounds(values, meta_data, nb_variable, bundles, selected_bun_indices, NR_POINTS,
                                   analysis_type,
                                   confound_names)

    # Significance testing with multiple correction of bundles
    if correct_mult_tract_comp:
        values_allp = []  # [subjects, NR_POINTS * nr_bundles, nb_variable]
        for s in meta_data["subject_id"]:
            values_subject = []
            for i, b_idx in enumerate(selected_bun_indices):
                values_subject += list(values[s][b_idx])  # concatenate all bundles
            values_allp.append(values_subject)

        alphaFWE, clusterFWE = get_corrected_alpha(values_allp, meta_data, analysis_type, subjects_A, subjects_B, alpha, bundles, nperm, b_idx)

    if FWE_method == "alphaFWE":
        results_df = pd.DataFrame(columns=["bundle", "alphaFWE", "min_pvalue", "t_value"])
    else:
        results_df = pd.DataFrame(columns=["bundle", "clusterFWE", "t_value"])

    for i, b_idx in enumerate(tqdm(selected_bun_indices)):
        if not correct_mult_tract_comp:
            values_allp = [values[s][b_idx] for s in subjects_A + subjects_B]  # [subjects, NR_POINTS]
            alphaFWE, clusterFWE = get_corrected_alpha(values_allp, meta_data, analysis_type, subjects_A, subjects_B,
                                                       alpha, bundles, nperm, b_idx)


        # Calc Hotelling T2 values for group analysis or multiple regression again a target
        stats = np.zeros(NR_POINTS)
        pvalues = np.zeros(NR_POINTS)
        fvalues = np.zeros(NR_POINTS)
        svalues = np.zeros([NR_POINTS, nb_variable, nb_variable])
        for jdx in range(NR_POINTS):
            if analysis_type == "group":
                values_controls = np.zeros([len(subjects_A), nb_variable])
                for k, sub_A in enumerate(subjects_A):
                    values_controls[k][:] = values[sub_A][b_idx][jdx]

                values_patients = np.zeros([len(subjects_B), nb_variable])
                for k, sub_B in enumerate(subjects_B):
                    values_patients[k][:] = values[sub_B][b_idx][jdx]

                stats[jdx], fvalues[jdx], pvalues[jdx], svalues[jdx] = hotelling_t2(values_controls, values_patients)
            else:
                values_subjects = np.zeros([len(subjects_A), nb_variable])
                for k, sub_A in enumerate(subjects_A):
                    values_subjects[k][:] = values[sub_A][b_idx][jdx]

                values_subjects_with_constant = sm.add_constant(values_subjects)

                est = sm.OLS(np.transpose(meta_data["target"].values), values_subjects_with_constant)
                est_result = est.fit()
                pvalues[jdx] = est_result.f_pvalue

        data = {"position": [],
                "fa": [],
                "group": [],
                "subject": []}

        for j, subject in enumerate(subjects_A + subjects_B):
            for var in range(nb_variable):
                for position in range(NR_POINTS):
                    data["position"].append(position)
                    data["subject"].append(subject)
                    data["fa"].append(values[subject][b_idx][position][var])
                    if subject in subjects_A:
                        data["group"].append("Group 0, var : " + str(var))
                    else:
                        data["group"].append("Group 1, var : " + str(var))

        # Plot
        ax = sns.lineplot(x="position", y="fa", data=data, ax=axes[i], hue="group")
        # units="subject", estimator=None, lw=1)  # each subject as single line

        ax.set(xlabel='position along tract', ylabel='metric')
        ax.set_title(bundles[b_idx])
        if analysis_type == "correlation" or hide_legend:
            ax.legend_.remove()
        elif analysis_type == "group" and i > 0:
            ax.legend_.remove()  # only show legend on first subplot

        # Plot significant areas
        if  show_detailed_p:
            ax2 = axes[i].twinx()
            ax2.bar(range(len(pvalues)), -np.log10(pvalues), color="gray", edgecolor="none", alpha=0.5)
            ax2.plot([0, NR_POINTS - 1], (-np.log10(alphaFWE),) * 2, color="red", linestyle=":")
            ax2.set(xlabel='position', ylabel='-log10(p)')

        if FWE_method == "alphaFWE":
            sig_areas_0_1 = get_significant_areas(pvalues, 1, alphaFWE)
        else:
            sig_areas_0_1 = get_significant_areas(pvalues, clusterFWE, alpha)

        sig_areas = sig_areas_0_1 * np.quantile(np.array(data["fa"]), 0.98)
        sig_areas[sig_areas == 0] = np.quantile(np.array(data["fa"]), 0.02)

        axes[i].plot(range(len(sig_areas)), sig_areas, color="red", linestyle=":")

        # Plot text
        results_df.at[i, "bundle"] = bundles[b_idx]
        if FWE_method == "alphaFWE":
            axes[i].annotate("alphaFWE:   {}".format(format_number(alphaFWE)),
                             (0, 0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top',
                             fontsize=10)
            axes[i].annotate("min p-value: {}".format(format_number(pvalues.min())),
                             (0, 0), (0, -45), xycoords='axes fraction', textcoords='offset points', va='top',
                             fontsize=10)
            results_df.at[i, "alphaFWE"] = format_number(alphaFWE)
            results_df.at[i, "min_pvalue"] = format_number(pvalues.min())
        else:
            axes[i].annotate("clusterFWE:   {}".format(clusterFWE),
                             (0, 0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top',
                             fontsize=10)
            results_df.at[i, "clusterFWE"] = clusterFWE

        stats_label = "t-value:      " if analysis_type == "group" else "corr.coeff.: "
        axes[i].annotate(stats_label + "   {}".format(format_number(stats[pvalues.argmin()])),
                         (0, 0), (0, -55), xycoords='axes fraction', textcoords='offset points', va='top',
                         fontsize=10)
        results_df.at[i, "t_value"] = format_number(stats[pvalues.argmin()])

        if plot_3D_type != "none":

            if plot_3D_type == "pval":
                metric = pvalues  # use this code if you want to plot the pvalues instead of sig_areas
            elif plot_3D_type == "sig_areas":
                metric = sig_areas_0_1

            bundle = bundles[b_idx]
            output_path_3D = output_path.split(".")[0] + "_" + bundle + "_3D.png"

            if tracking_dir == "auto":
                tracking_dir = tracking.get_tracking_folder_name("fixed_prob", False)

            if tracking_format == "tck":
                tracking_path = join(plot_3D_path, tracking_dir, bundle + ".tck")
            elif tracking_format == "vtk":
                tracking_path = join(plot_3D_path, tracking_dir, bundle + ".vtk")
            else:
                tracking_path = join(plot_3D_path, tracking_dir, bundle + ".trk")

            ending_path = join(plot_3D_path, "endings_segmentations", bundle + "_b.nii.gz")

            if not os.path.isfile(tracking_path):
                raise ValueError("Could not find: " + tracking_path)
            if not os.path.isfile(ending_path):
                raise ValueError("Could not find: " + ending_path)

            plot_utils_with_saving.plot_bundles_with_metric(tracking_path, atlas_path, ending_path, bundle,
                                                            metric, plot_3D_type, output_path)
  
        if y_range is not None:
            ax.set_ylim(y_range[0], y_range[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    if save_csv:
        results_df.to_csv(output_path + ".csv")


def two_floats(value):
    if value is None:
        values = None
    else:
        values = value.split()
        if len(values) != 2:
            raise argparse.ArgumentError
        values = [float(x) for x in values]
    return values


def main():
    parser = argparse.ArgumentParser(description="Test for significant differences and plot tractometry results.",
                                     epilog="Written by Jakob Wasserthal.")
    parser.add_argument("-i", metavar="subjects_file_path", dest="subjects_file",
                        help="txt file containing path of subjects", required=True)
    parser.add_argument("-o", metavar="plot_path", dest="output_path",
                        help="output png file containing plots", required=True)
    parser.add_argument("--mc", action="store_true", help="correct for multiple tract comparison",
                        default=False)
    parser.add_argument("--nperm", metavar="n", type=int, help="Number of permutations (default: 5000)",
                        default=5000)
    parser.add_argument("--alpha", metavar="a", type=float, help="The desired alpha level (default: 0.05)",
                        default=0.05)
    parser.add_argument("--plot3D", metavar="none|metric|pval", choices=["none", "metric", "pval"],
                        help="Generate streamline plots of each tract with FA ('metric') or significant areas "
                             "('pval') on top",
                        default="none")
    parser.add_argument("--save_csv", action="store_true", help="save results also as csv",
                        default=False)
    parser.add_argument("--tracking_dir", metavar="folder_name",
                        help="Set name of directory containing the tracking output (see same option for 'Tracking').",
                        default="auto")
    parser.add_argument("--tracking_format", metavar="tck|trk|vtk|trk_legacy", choices=["tck", "trk", "vtk", "trk_legacy"],
                        help="If using --plot3D you have to specify the format of the trackings which will get loaded."
                             "(default: trk_legacy)",
                        default="trk_legacy")
    parser.add_argument("--atlas_path", metavar="atlas_path", help="Atlas reference for the input tracks",
                        default="")
    parser.add_argument('--range', '-r', metavar='l u', default=None, type=two_floats,
                        help='Range of metric (y-axis) to plot. Specify lower (l) and upper (u) bound'
                             '(default: None)')
    parser.add_argument("--FWE-method", metavar="alphaFWE|clusterFWE", choices=["alphaFWE", "clusterFWE"],
                        help="Select FWE method",
                        default="alphaFWE")

    args = parser.parse_args()

    # Choose how to define significance: by corrected alphaFWE or by clusters of values smaller than uncorrected alpha
    # clusterFWE not recommended because misses highly significant areas just because cluster is not big enough.
    # FWE_method = alphaFWE | clusterFWE
    FWE_method = args.FWE_method

    # Show p-value for each position, not only significant areas
    show_detailed_p = True
    hide_legend = False
    show_color_bar = True  # colorbar on 3D plot (this is only relevant for metric plot; pval will never show bar)
    nperm = args.nperm

    # Significance testing with multiple correction for bundles
    correct_mult_tract_comp = args.mc

    print("Correcting for comparison of multiple tracts: {}".format(correct_mult_tract_comp))
    base_path, meta_data, nb_variable, selected_bundles, plot_3D_path = parse_subjects_file(args.subjects_file)

    if args.plot3D != "none":
        if plot_3D_path is None:
            raise ValueError("'# plot_3D=...' is not set in " + args.subjects_file)

    if meta_data.columns[1] == "group":
        analysis_type = "group"
        print("Doing group analysis.")
        print("Number of subjects:")
        print("  group 0: {}".format((meta_data["group"] == 0).sum()))
        print("  group 1: {}".format((meta_data["group"] == 1).sum()))
    elif meta_data.columns[1] == "target":
        analysis_type = "correlation"
        print("Doing correlation analysis.")
        nperm = int(nperm / 5)
        print("Using one fifth ({}) of the number of permutations for correlation analysis "
              "because this has longer runtime".format(nperm))
        print("Number of subjects: {}".format(len(meta_data)))
    else:
        raise ValueError("Invalid second column header (only 'group' or 'target' allowed)")


    # all_bundles = dataset_specific_utils.get_bundle_names("All_tractometry")[1:]
    all_bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right',
                   'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left', 'ILF_right',
                   'MCP',
                   'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right', 'SLF_I_left',
                   'SLF_I_right',
                   'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right', 'STR_left', 'STR_right', 'UF_left',
                   'UF_right', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left', 'T_PREM_right', 'T_PREC_left',
                   'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left', 'T_PAR_right', 'T_OCC_left',
                   'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left', 'ST_PREF_right', 'ST_PREM_left',
                   'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left', 'ST_POSTC_right', 'ST_PAR_left',
                   'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    NR_POINTS = np.loadtxt(base_path.replace("SUBJECT_ID", meta_data["subject_id"][0]) + "_0.csv", delimiter=";",
                           skiprows=1).transpose().shape[1]

    values = {}
    for subject in meta_data["subject_id"]:
        raw = np.zeros([len(all_bundles), NR_POINTS, nb_variable])
        for k in range(nb_variable):
            raw[:, :, k] = np.loadtxt(base_path.replace("SUBJECT_ID", subject) + "_" + str(k) + ".csv", delimiter=";",
                                      skiprows=1).transpose()

        values[subject] = raw

    plot_tractometry_with_pvalue(values, meta_data, nb_variable, all_bundles, selected_bundles, args.output_path,
                                 args.alpha, FWE_method, analysis_type, correct_mult_tract_comp,
                                 show_detailed_p, nperm=nperm, hide_legend=hide_legend,
                                 plot_3D_path=plot_3D_path, plot_3D_type=args.plot3D,
                                 tracking_format=args.tracking_format, tracking_dir=args.tracking_dir,
                                 atlas_path=args.atlas_path, show_color_bar=show_color_bar, save_csv=args.save_csv,
                                 y_range=args.range)


if __name__ == '__main__':
    main()
