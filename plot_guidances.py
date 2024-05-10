import os
from src.data.text import load_json
import re
from tqdm import tqdm
from numpy import number
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_PATH = 'experiments/clean-motionfix' 

def get_metrs_n_guids(subpath, metr, set_to_eval='test'):
    # path_for_exps = f'{BASE_PATH}/{subpath}'
    path_for_exps = subpath
    # import ipdb; ipdb.set_trace()
    loxps=[ f.path for f in os.scandir(path_for_exps) if f.is_dir()]
    expname = '__'.join(path_for_exps.split('/')[-3:-1])
    print(f'=========={expname}=========')
    if set_to_eval == 'test':
        set_str = ''
    else:
        set_str = '_val'
    guidance_tnm = []
    guidance_m = []
    s2t_avgr_batch = []
    t2t_avgr_batch = []
    s2t_avgr_all = []
    t2t_avgr_all = []
    not_found = 0
    not_found_paths = []
    for x in tqdm(loxps):
        gd_comb = x.split('/')[-1]
        numbers = re.findall(r'\d+\.\d+', gd_comb)
        try:
            data_batch=load_json(x+f'/batches_res{set_str}.json')
            data_all=load_json(x+f'/all_res{set_str}.json')
        except:
            not_found += 1
            not_found_paths.append(x)
            continue
        guidance_tnm.append(float(numbers[0]))
        guidance_m.append(float(numbers[1]))
        s2t_avgr_batch.append(float(data_batch[f'{metr}_s2t']))
        t2t_avgr_batch.append(float(data_batch[f'{metr}']))
        s2t_avgr_all.append(float(data_all[f'{metr}_s2t']))
        t2t_avgr_all.append(float(data_all[f'{metr}']))
    print(f'Number of paths of not found: {not_found}')
    # import ipdb;ipdb.set_trace()
    if not_found > 0:
        print(f'List of paths that are not found\n{not_found_paths}')
    return guidance_tnm, guidance_m, s2t_avgr_batch, t2t_avgr_batch, s2t_avgr_all, t2t_avgr_all


def plot_2d_3d_plot(x, y, z, xname, yname, 
                    invert_size=False, metric='AvgR',
                    name=''):
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    norm = Normalize(vmin=min(z), vmax=max(z))
    z_normalized = norm(z)  # Apply normalization
    if invert_size:
        # Invert the size scaling: bigger numbers get smaller circles
        z_area = [(1 - z_size + 0.1) ** 2 * 1000 for z_size in z_normalized]
    else:
        # Normal size scaling: bigger numbers get bigger circles
        z_area = [(z_size + 0.1) ** 2 * 1000 for z_size in z_normalized]
    
    colors = plt.cm.cividis(z_normalized)
    # Plot each point with an area proportional to z value
    # import ipdb;ipdb.set_trace()
    scatter = ax.scatter(x, y, s=z_area, c=colors, alpha=0.8,
                        #  norm=norm,
                         marker='o',
                         edgecolor='k', linewidth=0.5)  # Use 'o' as the marker for circles
    # Annotate each circle with its z value
    for i, txt in enumerate(z):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", 
                    xytext=(0,15), ha='center')
    ax.set_xlabel(xname, fontsize=20)
    ax.set_ylabel(yname,fontsize=20)
    title_name = name.split('|')
    title_name = '\n'.join(title_name)
    ax.set_title(f'info: {title_name}', 
                 fontsize=11,  pad=20)
    # cbar = plt.colorbar(scatter, ax=ax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis'), ax=ax)
    
    cbar.set_label(f'{metric}',fontsize=20)
    # Set the ticks of the colorbar to match the original z values
    # tick_positions = np.linspace(0, 1, 5)
    # tick_labels = np.linspace(min(z), max(z), 5)
    # cbar.set_ticks(tick_positions)  # Position of ticks
    # cbar.set_ticklabels(np.round(tick_labels, 2))  # Text of ticks
    tick_positions = np.linspace(min(z), max(z), 5)
    cbar.set_ticks(tick_positions)  # Position of ticks
    cbar.set_ticklabels(np.round(tick_positions, 2))  # Text of ticks
    cbar.ax.tick_params(labelsize=10)
    # Set the ticks of the colorbar to show five evenly spaced values
    
    # Adjust the plot's axes limits to add more space
    x_margin = 0.2  # Additional x margin
    y_margin = 0.2    # Additional y margin
    ax.set_xlim(min(x) - x_margin, max(x) + x_margin)
    ax.set_ylim(min(y) - y_margin, max(y) + y_margin)
    print(f"saved in : guid_grids/{metric}{name.split('-')[0]}.png")
    plt.savefig(f"./guid_grids/{metric}{name.split('-')[0]}.png")
    # plt.savefig(f'./{metric}_{name}.pdf')

def main(path_samples, metric, set_for_eval):
    g_tnm, g_m,  s2t_bt, t2t_bt, s2t_all, t2t_all = get_metrs_n_guids(path_samples, 
                                                                      metr=metric,
                                                                      set_to_eval=set_for_eval)
    # inds = [idx for idx, (a,b) in enumerate(zip(g_m, g_tnm)) if a<4 and b<4]
    # g_m = [el for ii, el in enumerate(g_m) if ii in inds]
    # g_tnm = [el for ii, el in enumerate(g_tnm) if ii in inds]
    # s2t_bt = [el for ii, el in enumerate(s2t_bt) if ii in inds]
    # t2t_bt = [el for ii, el in enumerate(t2t_bt) if ii in inds]
    # import ipdb; ipdb.set_trace()
    path_samples = path_samples.rstrip('/')
    exp_alias = '__'.join(path_samples.split('/')[-3:-1])
    samples_name = path_samples.split('/')[-1]
    # import ipdb;ipdb.set_trace()
    extra_name = ''
    if metric in ['AvgR', 'MedR']:
        higher_is_better = False # eg R@1
    else:
        higher_is_better = True # eg R@1

    order = False if higher_is_better else True
    plot_2d_3d_plot(g_tnm, g_m, t2t_bt,
                    '$g_{text}^{motion}$', '$g^{motion}$',
                    metric=metric,
                    name=f'Gen2Target-{set_for_eval}|{samples_name}|{exp_alias}',
                    invert_size=order)
    plot_2d_3d_plot(g_tnm, g_m, s2t_bt,
                    '$g_{text}^{motion}$', '$g^{motion}$',
                    metric=metric,
                    name=f'Source2Gen-{set_for_eval}__{samples_name}__{exp_alias}',
                    invert_size=order)


if __name__ == '__main__':
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Add the arguments
    parser.add_argument('-p', '--path', type=str, help='The path to process')
    parser.add_argument('-m', '--metric', type=str, help='The string to process')
    parser.add_argument('--set',required=False,
                        default='test',
                        type=str, help='The string to process')
    # Parse the arguments
    args = parser.parse_args()

# subpath = 'bodilex_hml3d_sinc_synth/30-35-35_bs128_300ts_clip77_with_zeros_source/steps_1000_bodilex_noise_last'
# subpath = 'hml3d_sinc_synth/50-50_bs128_300ts_clip77_with_zeros_source/steps_1000_sinc_synth_noise_last'
    # subpath = 'bodilex_hml3d/50-50_bs128_300ts_clip77_with_zeros_source/steps_1000_bodilex_noise_last'
    # metric = 'R@1'
    metric_name = args.metric
    path_to_samples = args.path
    set_for_eval = args.set
    main(path_to_samples, metric_name, set_for_eval=set_for_eval)
