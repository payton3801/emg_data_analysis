import logging
from copy import deepcopy
import pandas as pd
import matplotlib.cm as colormap
import matplotlib.pyplot as plt
import numpy as np
from analysis.utils.core import BaseAnalysis, extract_group_df, set_trial_dataframes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.metrics import r2_score
from snel_toolkit.datasets.base import DataWrangler

# sklearn_logger = logging.getLogger("sklearnex")
# sklearn_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class PSTH(BaseAnalysis):
    def __init__(self, cfg_node, all_ds):
        super().__init__(cfg_node, all_ds)

    def get_preprocess_params_from_cfg(self):
        sub_key = "PREPROCESS"
        params = self.get_params_from_cfg(sub_key)
        required_keys = ["EMG_LF_CUTOFF_FREQ"]
        self.check_required_keys(params, required_keys)
        return params

    def get_collect_params_from_cfg(self):
        sub_key = "COLLECT"
        params = self.get_params_from_cfg(sub_key)

        required_keys = ["DS_NAME", "GROUP_FIELD"]
        self.check_required_keys(params, required_keys)
        return params

    def get_select_params_from_cfg(self):
        sub_key = "SELECT"
        params = self.get_params_from_cfg(sub_key)

        required_keys = [
            "DS_NAMES",
            "ALIGN_NAMES",
            "ALIGN_FIELDS",
            "ALIGN_RANGES",
            "IGNORE_TRIAL_NUMS",
        ]
        self.check_required_keys(params, required_keys)
        return params

    def get_post_params_from_cfg(self):
        #sub_key = "POSTPROCESS"
        #params = self.get_params_from_cfg(sub_key)
        #required_keys = []
        #self.check_required_keys(params, required_keys)
        return None
    
    def postprocess(self):
        return None


    def preprocess(self, EMG_FIELD, EMG_LF_CUTOFF_FREQ, 
                   APPLY_PCA, APPLY_PCA_FIELDS, NUM_PCS):
        logger.info("Running preprocessing.")
        ds_keys = list(self.all_ds.keys())
        smth_emg_name = f"{EMG_FIELD}_lf{EMG_LF_CUTOFF_FREQ}"
        for ds_key in ds_keys:
            # perform smoothing of EMG
            if smth_emg_name in self.all_ds[ds_key].data.columns.levels[0].tolist():
                bypass = True
            else:
                bypass = False
            if not bypass:
                self.all_ds[ds_key].smooth_cts(
                    signal_type="model_emg",
                    filt_type="butter",
                    crit_freq=EMG_LF_CUTOFF_FREQ,
                    btype="low",
                    overwrite=False,
                    name=f"lf{EMG_LF_CUTOFF_FREQ}",
                )
            else:
                logger.info(f"{smth_emg_name} already exists")
            if APPLY_PCA:
                for pca_field in APPLY_PCA_FIELDS:
                    new_pca_name = f"{pca_field}_PCs"
                    if new_pca_name in self.all_ds[ds_key].data.columns.levels[0].tolist():
                        bypass = True
                    else:
                        bypass = False
                    if not bypass:
                        # create nan mask to block from PCA fitting
                        nan_mask = self.all_ds[ds_key].data[pca_field].isnull().any(axis=1)
                        # Standardize input data and fit and apply PCA
                        
                        PCA_MODEL = PCA(NUM_PCS)
                        pcs = PCA_MODEL.fit_transform(StandardScaler().fit_transform(self.all_ds[ds_key].data[pca_field].loc[~nan_mask,:]))
                        # create a new dataframe to hold pca output with index like the input
                        
                        pcs_df = pd.DataFrame().reindex_like(self.all_ds[ds_key].data[pca_field])
                        if not (NUM_PCS is None):
                            pcs_df = pcs_df.iloc[:, :NUM_PCS]
                        # fill the dataframe with output of pca
                        pcs_df.loc[~nan_mask,:] = pcs

                        # extract number of PCs to create name for each dimension
                        num_pcs = pcs_df.shape[1]

                        # create multiindex to enable concatenation to original dataframe
                        midx = pd.MultiIndex.from_tuples([(new_pca_name, f"PC{(i+1):02}") for i in range(num_pcs)])

                        # overwrite column names of PC df with multindex
                        pcs_df.columns = midx

                        # concatenate pcs to original dataframe and overwrite in Dataset
                        self.all_ds[ds_key].data = pd.concat([self.all_ds[ds_key].data, pcs_df], axis=1)
                    else:
                        logger.info(f"{new_pca_name} already exists")
        # set flag to True
        self.preprocessed = True

    def select(
        self, DS_NAMES, ALIGN_NAMES, ALIGN_FIELDS, ALIGN_RANGES, IGNORE_TRIAL_NUMS, ALLOW_NANS=False
    ):
        logger.info("Running selection.")
        assert self.preprocessed
        self.all_dw = {}
        self.dw_keys = DS_NAMES
        for ds_name in DS_NAMES:
            dataset = self.all_ds[ds_name]
            self.all_dw[ds_name] = DataWrangler(dataset)
            set_trial_dataframes(
                self.all_dw[ds_name],
                ALIGN_NAMES,
                ALIGN_FIELDS,
                ALIGN_RANGES,
                IGNORE_TRIAL_NUMS,
                ALLOW_NANS
            )
        # set flag to True
        self.selected = True

    def collect(self, DS_NAME, GROUP_FIELD):
        logger.info("Running collection.")
        assert self.selected and self.preprocessed
        # get all groups
        self.dw_key = DS_NAME
        self.group_field = GROUP_FIELD
        dw = self.all_dw[self.dw_key]
        GROUP_IDS = dw.get_groups(GROUP_FIELD)
        align_names = list(dw._d._t_dfs.keys())

        def prepare_aligned_data_obj(dw, ALIGN_NAMES, GROUP_FIELD, GROUP_IDS):
            aligned_data_obj = {}
            for align_name in ALIGN_NAMES:
                # set which aligned trial dataframe to plot
                dw.set_trial_data(align_name)
                # for each group id in passed list, extract group data
                aligned_data_obj[align_name] = {}
                aligned_data_obj[align_name]["dfs"] = {}
                for i, group_id in enumerate(GROUP_IDS):
                    logger.info(f"Getting group data for {GROUP_FIELD} | {group_id} |")
                    group_df = dw.get_group_data(GROUP_FIELD, group_id)
                    a_df = extract_group_df(group_df, piv_col_name="trial_id")
                    aligned_data_obj[align_name]["dfs"][group_id] = a_df
                    # on first iteration save the appropriate time vector for plotting
                    # also save the names of the channels that are being plotted
                    if i == 0:
                        aligned_data_obj[align_name][
                            "time_vector"
                        ] = group_df.align_time.astype("timedelta64[ms]").unique()

            return aligned_data_obj

        ad_obj = prepare_aligned_data_obj(dw, align_names, GROUP_FIELD, GROUP_IDS)
        self.data = ad_obj

        self.collected = True

    def main(self, cfg_node):
        logger.info("Running main analysis function.")
        a_key = "ANALYSIS_PARAMS"
        p_key = "PLOT_PSTH"
        f_key = "FIGURE"

        # plot params
        PLOT_FIELD = cfg_node[a_key][p_key]["FIELD"]
        FIELDNAMES = cfg_node[a_key][p_key]["FIELDNAMES"]
        GROUP_IDS = cfg_node[a_key][p_key]["GROUP_IDS"]
        SEGMENT_SPACING_MS = cfg_node[a_key][p_key]["SEGMENT_SPACING_MS"]
        COLOR_TYPE = cfg_node[a_key][p_key]["COLOR_TYPE"]
        ALPHA_MEAN = cfg_node[a_key][p_key]["ALPHA_MEAN"]
        ALPHA_SEM = cfg_node[a_key][p_key]["ALPHA_SEM"]
        ALPHA_ST = cfg_node[a_key][p_key]["ALPHA_ST"]
        LW_MEAN = cfg_node[a_key][p_key]["LINEWIDTH_MEAN"]
        LW_ST = cfg_node[a_key][p_key]["LINEWIDTH_ST"]
        AVERAGE = cfg_node[a_key][p_key]["AVERAGE"]
        STD_COLOR = cfg_node[a_key][p_key]["STD_COLOR"]
        STD_FIELD = cfg_node[a_key][p_key]["STD_FIELD"]
        # figure params
        DPI = cfg_node[f_key]["DPI"]
        MAX_PLOTS_PER_FIG = cfg_node[f_key]["MAX_PLOTS_PER_FIG"]
        SCALE_X = cfg_node[f_key]["SCALE_X"]
        SCALE_Y = cfg_node[f_key]["SCALE_Y"]
        NCOLS = cfg_node[f_key]["NCOLS"]

        def plot_single_trials(ax, t_vec, data_txb, color, std_color, line_colors=None, **kwargs):
            if std_color:
                for i, l in enumerate(data_txb.values.T):
                    ax.plot(t_vec, l, color=line_colors[i], **kwargs)
            else:
                ax.plot(t_vec, data_txb, color=color, **kwargs)

        def plot_psth(
            ax, t_vec, data_txb, color, mean_alpha=0.9, sem_alpha=0.2, **kwargs
        ):
            cond_avg = np.nanmean(data_txb, axis=1)
            cond_sem = np.nanstd(data_txb, axis=1) / np.sqrt(data_txb.shape[1])
            ax.plot(t_vec, cond_avg, color=color, alpha=mean_alpha, **kwargs)
            ax.fill_between(
                t_vec,
                cond_avg - cond_sem,
                cond_avg + cond_sem,
                color=color,
                alpha=sem_alpha,
            )

        # --- compute size of plot
        nsegs = len(self.data.keys())
        seg_keys = list(self.data.keys())
        ALL_GROUP_IDS = list(self.data[seg_keys[0]]["dfs"].keys())
        if GROUP_IDS is None:
            logger.info("No group ids passed. Returning all group_ids")
            GROUP_IDS = ALL_GROUP_IDS

        pv_ad = self.data[seg_keys[0]]["dfs"][GROUP_IDS[0]]
        t_ix = pv_ad[PLOT_FIELD].columns.levels[1][0]
        all_fieldnames = (
            pv_ad[PLOT_FIELD]
            .swaplevel(axis=1)
            .sort_index(axis=1)[t_ix]
            .columns.tolist()
        )

        if FIELDNAMES is None:
            logger.info(
                f"No fieldnames passed. Returning all fieldnames for {PLOT_FIELD}"
            )
            FIELDNAMES = all_fieldnames
        logger.info(f"Fieldnames extracted: {FIELDNAMES}")
        n_total_plots = len(FIELDNAMES)
        fnames = deepcopy(FIELDNAMES)
        if n_total_plots < MAX_PLOTS_PER_FIG:
            nrows = np.ceil(n_total_plots / NCOLS).astype(int)
        else:
            nrows = np.floor(MAX_PLOTS_PER_FIG / NCOLS).astype(int)

        n_plots_per_fig = NCOLS * nrows
        n_figs = np.ceil(n_total_plots / (n_plots_per_fig)).astype(int)
        logger.info(f"Creating {n_figs} figure(s).")
        logger.info(f"Figures will have {NCOLS} column(s) and {nrows} row(s).")
        logger.info(f"Each subplot will show {nsegs} segment(s).")

        FIGSIZE = (NCOLS * SCALE_X, nrows * SCALE_Y)
        DO_COMPARE = cfg_node[a_key][p_key]["COMPARISON"]["DO_COMPARE"]
        COMPUTE_PSTH_R2 = False  # starts as False, can be flipped to True via config
        if DO_COMPARE:
            COMPARE_FIELDS = cfg_node[a_key][p_key]["COMPARISON"]["FIELDS"]
            N_COMPARE = len(COMPARE_FIELDS) + 1
            # double number of columns (plot, compare)
            NCOLS *= N_COMPARE
            for COMPARE_FIELD in COMPARE_FIELDS:
                comp_fieldnames = (
                    self.data[seg_keys[0]]["dfs"][GROUP_IDS[0]][COMPARE_FIELD]
                    .swaplevel(axis=1)
                    .sort_index(axis=1)[t_ix]
                    .columns.tolist()
                )

                def check_subset(base_list, sub_list):
                    # all elements of sub_list must be in base list
                    return all([x in base_list for x in sub_list])

                assert check_subset(
                    comp_fieldnames, FIELDNAMES
                ), f"'{COMPARE_FIELD}' must have all fieldnames passed from '{PLOT_FIELD}' \
                        {comp_fieldnames} || {FIELDNAMES}"
            COMPUTE_PSTH_R2 = cfg_node[a_key][p_key]["COMPARISON"][
                "COMPUTE_PSTH_R2"
            ]
            if COMPUTE_PSTH_R2:
                plt_data_for_r2 = []
                comp_data_for_r2 = {}
                psth_r2_vals = {}
                for COMPARE_FIELD in COMPARE_FIELDS:
                    comp_data_for_r2[COMPARE_FIELD] = []
                    psth_r2_vals[COMPARE_FIELD] = []

        else:
            N_COMPARE = 1
        # --- determine coloring for groups
        if COLOR_TYPE == "location":
            c_field = "tgt_loc"
            cm_op = loc_cm_op
            cm = colormap.hsv
        elif COLOR_TYPE == "object":
            c_field = "obj_id"
            cm_op = obj_cm_op
            cm = colormap.Dark2
        COLOR_DICT = build_color_dict(
            self.all_dw[self.dw_key], self.group_field, c_field, cm, cm_op
        )

        # --- plotting loop
        for i_fig in range(n_figs):
            plt_count = 0  # counter to track which subplot to use
            logger.info(f"Generating figure {i_fig+1}/{n_figs}")
            # create subplot figure
            fig, axs = plt.subplots(
                nrows, NCOLS, sharey="none", sharex="all", figsize=FIGSIZE, dpi=DPI
            )
            axs = axs.flatten()

            

            # for each plot in figure
            for i_plot in range(n_plots_per_fig):
                # get fieldname
                try:
                    fname = fnames.pop(0)
                except IndexError:  # if out of things to plot break loop
                    break
                t_offset = 0
                # for each segment
                for i_seg, seg_key in enumerate(seg_keys):
                    t_vec = self.data[seg_key]["time_vector"]
                    offset_t_vec = t_vec - t_vec[0] + t_offset
                    # for each condition
                    for i_group, group_id in enumerate(GROUP_IDS):

                        def get_field_df(group_df, field, fieldname):
                            a_data = group_df[field][fieldname]
                            return a_data

                        def plot_data(axs, ix, t_vec, a_df, field, fieldname, color, std_color, line_colors):

                            if AVERAGE:
                                plot_psth(
                                    axs[ix],
                                    t_vec,
                                    a_df,
                                    color=color,
                                    mean_alpha=ALPHA_MEAN,
                                    sem_alpha=ALPHA_SEM,
                                    linewidth=LW_MEAN,)
                            else:
                                plot_single_trials(
                                    axs[ix],
                                    t_vec,
                                    a_df,
                                    color=color,
                                    std_color=std_color,
                                    line_colors=line_colors,
                                    alpha=ALPHA_ST,
                                    linewidth=LW_ST,
                                )
                            if DO_COMPARE or ix == 0:
                                axs[ix].set_title(f"{field}:{fieldname}", fontsize=6)
                            else:
                                axs[ix].set_title(f"{fieldname}")

                        plt_df = get_field_df(
                            self.data[seg_key]["dfs"][group_id], PLOT_FIELD, fname
                        )
                        line_colors = []
                        if STD_COLOR:
                            stds = np.std(plt_df, axis=1)
                            max_idx = np.argmax(stds)
                            
                            norm = plt.Normalize(plt_df.values[max_idx, :].min(), 
                                                 plt_df.values[max_idx, :].max())
                            cmap = plt.get_cmap('viridis')
                            line_colors = cmap(norm(plt_df.values[max_idx, :]))

                        plot_data(
                            axs,
                            plt_count,
                            offset_t_vec,
                            plt_df,
                            PLOT_FIELD,
                            fname,
                            COLOR_DICT[group_id],
                            std_color=STD_COLOR,
                            line_colors=line_colors
                        )
                        if AVERAGE and COMPUTE_PSTH_R2:
                            plt_data_for_r2.append(np.nanmean(plt_df, axis=1))
                        elif COMPUTE_PSTH_R2:
                            plt_data_for_r2.append(plt_df)
                        if DO_COMPARE:
                            tmp_plt_count = deepcopy(plt_count)
                            for COMPARE_FIELD in COMPARE_FIELDS:
                                tmp_plt_count += 1
                                # if last iteration, then match y-axes of PLOT_FIELD
                                if i_group == len(GROUP_IDS) - 1:
                                    axs[tmp_plt_count].set_ylim(
                                        axs[tmp_plt_count - 1].get_ylim()
                                    )
                                    # axs[tmp_plt_count].get_shared_y_axes().join(
                                    #     axs[tmp_plt_count], axs[tmp_plt_count - 1]
                                    # )
                                comp_df = get_field_df(
                                    self.data[seg_key]["dfs"][group_id],
                                    COMPARE_FIELD,
                                    fname,
                                )
                                plot_data(
                                    axs,
                                    tmp_plt_count,
                                    offset_t_vec,
                                    comp_df,
                                    COMPARE_FIELD,
                                    fname,
                                    COLOR_DICT[group_id],
                                    STD_COLOR,
                                    line_colors
                                )
                                # -- compute and report PSTH R^2
                                if AVERAGE and COMPUTE_PSTH_R2:
                                    comp_data_for_r2[COMPARE_FIELD].append(
                                        np.nanmean(comp_df, axis=1)
                                    )
                                    if (
                                        i_group == len(GROUP_IDS) - 1
                                        and i_seg == len(seg_keys) - 1
                                    ):
                                        psth_r2 = r2_score(
                                            np.concatenate(plt_data_for_r2),
                                            np.concatenate(
                                                comp_data_for_r2[COMPARE_FIELD]
                                            ),
                                        )
                                        axs[tmp_plt_count].text(
                                            offset_t_vec[-1] * 0.6,
                                            axs[tmp_plt_count].get_ylim()[1] * 0.8,
                                            f"$R^2$:{psth_r2:.2f}",
                                        )
                                        psth_r2_vals[COMPARE_FIELD].append(psth_r2)
                                elif COMPUTE_PSTH_R2:
                                    print('we in here')
                                    comp_data_for_r2[COMPARE_FIELD].append(
                                        comp_df
                                    )
                                    if (
                                        i_group == len(GROUP_IDS) - 1
                                        and i_seg == len(seg_keys) - 1
                                    ):
                                        psth_r2 = r2_score(
                                            np.concatenate(plt_data_for_r2),
                                            np.concatenate(
                                                comp_data_for_r2[COMPARE_FIELD]
                                            ),
                                        )
                                        axs[tmp_plt_count].text(
                                            offset_t_vec[-1] * 0.6,
                                            axs[tmp_plt_count].get_ylim()[1] * 0.8,
                                            f"$R^2$:{psth_r2:.2f}",
                                        )
                                        psth_r2_vals[COMPARE_FIELD].append(psth_r2)
                                axs[tmp_plt_count].get_yaxis().set_visible(False)
                                axs[tmp_plt_count].spines["left"].set_visible(False)
                            tmp_plt_count = plt_count
                    # shift offset
                    t_offset += offset_t_vec[-1] + SEGMENT_SPACING_MS
                # iterate to next channel
                plt_count += N_COMPARE
            for ax in axs:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            if COMPUTE_PSTH_R2:
                if len(GROUP_IDS) < len(ALL_GROUP_IDS):
                    logger.warn(
                        "PSTH R^2 computed on only as subset of the conditions: "
                        f"({len(GROUP_IDS)}/{len(ALL_GROUP_IDS)})"
                    )
                psth_r2_vals["ref_fieldname"] = PLOT_FIELD

                self.results = {}
                self.results["psth_r2"] = psth_r2_vals
            fig.tight_layout()


def build_color_dict(dw, GROUP_FIELD, COLOR_FIELD, cm, colormap_op=None):
    assert colormap_op is not None, "Must pass colormap operation to define color."
    group_ids = dw.get_groups(GROUP_FIELD)
    color_dict = {}
    for group_id in group_ids:
        # add color to color dict for group id
        color_dict[group_id] = colormap_op(dw, group_id, cm, GROUP_FIELD, COLOR_FIELD)
    return color_dict


def loc_cm_op(dw, group_id, cm, GROUP_FIELD, COLOR_FIELD):
    tgt_loc = dw._d.ti[COLOR_FIELD][dw._d.ti[GROUP_FIELD] == group_id].iloc[0] + 180
    return cm(tgt_loc / 360)


def obj_cm_op(dw, group_id, cm, GROUP_FIELD, COLOR_FIELD):
    obj_id = dw._d.ti[COLOR_FIELD][dw._d.ti[GROUP_FIELD] == group_id].iloc[0]
    return cm(obj_id)
