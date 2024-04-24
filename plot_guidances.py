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

def get_metrs_n_guids(subpath, metr):
    # path_for_exps = f'{BASE_PATH}/{subpath}'
    path_for_exps = subpath
    loxps=[ f.path for f in os.scandir(path_for_exps) if f.is_dir()]
    expname = '__'.join(path_for_exps.split('/')[-3:-1])
    print('=========={expname}=========')

    guidance_tnm = []
    guidance_m = []
    s2t_avgr_batch = []
    t2t_avgr_batch = []
    s2t_avgr_all = []
    t2t_avgr_all = []

    for x in tqdm(loxps):
        gd_comb = x.split('/')[-1]
        numbers = re.findall(r'\d+\.\d+', gd_comb)
        print(numbers)
        guidance_tnm.append(float(numbers[0]))
        guidance_m.append(float(numbers[1]))

        data_batch=load_json(x+'/batches_res.json')
        data_all=load_json(x+'/all_res.json')
        s2t_avgr_batch.append(float(data_batch[f'{metr}_s2t']))
        t2t_avgr_batch.append(float(data_batch[f'{metr}']))
        s2t_avgr_all.append(float(data_all[f'{metr}_s2t']))
        t2t_avgr_all.append(float(data_all[f'{metr}']))
        print('---')
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
    scatter = ax.scatter(x, y, s=z_area, c=colors, alpha=0.8,
                         norm=norm, marker='o',
                         edgecolor='k', linewidth=0.5)  # Use 'o' as the marker for circles
    # Annotate each circle with its z value
    for i, txt in enumerate(z):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", 
                    xytext=(0,15), ha='center')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(f'Plot of guidances and {metric}')
    # cbar = plt.colorbar(scatter, ax=ax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis'), ax=ax)
    
    cbar.set_label(f'{metric}')
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
    plt.savefig(f'./{metric}{name}.png')
    plt.savefig(f'./{metric}{name}.pdf')

def main(path_samples, metric):
    g_tnm, g_m,  s2t_bt, t2t_bt, s2t_all, t2t_all = get_metrs_n_guids(path_samples, 
                                                                      metr=metric)
    extra_name = ''
    if metric in ['AvgR', 'MedR']:
        higher_is_better = False # eg R@1
    else:
        higher_is_better = True # eg R@1

    order = False if higher_is_better else True
    plot_2d_3d_plot(g_tnm, g_m, t2t_bt,
                    '$g_{text}^{motion}$', '$g^{motion}$',
                    metric=metric,
                    name=extra_name,
                    invert_size=order)

if __name__ == '__main__':
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Process some inputs.")

    # Add the arguments
    parser.add_argument('-p', '--path', type=str, help='The path to process')
    parser.add_argument('-m', '--metric', type=str, help='The string to process')

    # Parse the arguments
    args = parser.parse_args()

# subpath = 'bodilex_hml3d_sinc_synth/30-35-35_bs128_300ts_clip77_with_zeros_source/steps_1000_bodilex_noise_last'
# subpath = 'hml3d_sinc_synth/50-50_bs128_300ts_clip77_with_zeros_source/steps_1000_sinc_synth_noise_last'
    # subpath = 'bodilex_hml3d/50-50_bs128_300ts_clip77_with_zeros_source/steps_1000_bodilex_noise_last'
    # metric = 'R@1'
    metric_name = args.metric
    path_to_samples = args.path
    main(path_to_samples, metric_name)