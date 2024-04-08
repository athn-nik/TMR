import sys
from rich import print
import time
import subprocess
import re
import argparse

def run(cmd):
    print(f"Executing: {cmd}")
    x = subprocess.run(cmd)

def get_guidances(s=1, e=3, no=3, t2m=False):
    import itertools
    import numpy as np
    if t2m:
        gd_text = np.linspace(s, e, no, endpoint=True)
        all_combs = [round(c,2) for c in list(gd_text)]
    else:
        gd_motion = np.linspace(s, e, no, endpoint=True)
        gd_text_motion = np.linspace(s, e, no, endpoint=True)

        all_combinations = list(itertools.product(gd_motion, gd_text_motion))
        all_combs = [(round(c[0],2), round(c[1],2)) for c in all_combinations]

    return all_combs

def main_loop(command, exp_paths,
              guidance_vals,
              init_from, data):

    cmd_no=0
    cmd_sample = command
    from itertools import product
    exp_grid = list(product(exp_paths,
                            guidance_vals,
                            init_from,
                            data))
    print("Number of different experiments is:", len(exp_grid))
    print('---------------------------------------------------')
    ckt_name = 'last'
    if data[0] != 'hml3d':
        arg1 = 'guidance_scale_text_n_motion'
        arg0 = 'guidance_scale_motion'
        t2m = False
    else:
        arg0 = 'guidance_scale_text'
        arg1 = 'guidance_scale_motion'
        t2m = True
    for fd, gd, in_lat, data_type in exp_grid:
        cur_cmd = list(cmd_train)
        idx_of_exp = cur_cmd.index("FOLDER")
        cur_cmd[idx_of_exp] = str(fd)
        if t2m:
            list_of_args = ' '.join([f"init_from={in_lat}",
                                 f"ckpt_name={ckt_name}",
                                 f"{arg1}={gd[1]}",
                                 f"{arg0}={gd[0]}",
                                 f"data={data_type}"])
        else:
            list_of_args = ' '.join([f"init_from={in_lat}",
                                 f"ckpt_name={ckt_name}",
                                 f"{arg1}={gd[1]}",
                                 f"{arg0}={gd[0]}",
                                 f"data={data_type}"])
        cur_cmd.extend([list_of_args])
        run(cur_cmd)
        time.sleep(0.2)
        cmd_no += 1
    import ipdb;ipdb.set_trace()
if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument('--ds', required=True, type=str,
                        help='dataset')
    parser.add_argument('--bid', required=False, default=30, type=int,
                        help='bid money for cluster')
    parser.add_argument('--extras', required=False, default='', type=str, help='args hydra')

    args = parser.parse_args()
    bid_for_exp = args.bid
    datasets = args.ds
    extras_args = args.extras
    import ipdb; ipdb.set_trace()
    start_index = extras_args.find("samples_path=") + len("samples_path=")
    end_index = extras_args.find(" ", start_index)
    if end_index == -1:  # In case there is no space after the target substring
        base_dir_of_samples = extras_args[start_index:]
    else:
        base_dir_of_samples = extras_args[start_index:end_index]
    samples_dirs_list = [os.path.join(base_dir_of_samples, d) for d in os.scandir(base_dir_of_samples)
                         if d.is_dir()]
    print('The current folders are---->\n', '\n'.join(subdirs))
    print('---------------------------------------------------')
    assert datasets in ['bodilex', 'sinc_synth', 'hml3d']
    parser = argparse.ArgumentParser()
    cmd_eval = ['python', 'cluster/single_run.py',
                 '--folder', 'FOLDER',
                 '--bid', '20',
                 '--extras', extras_args]
    init_from = ['noise', 'source']
    data = [datasets]
    main_loop(cmd_eval, samples_dirs_list,,
              data)
    # python evaluate_cluster.py --extras "run_dir=outputs/tmr_humanml3d_amass_feats dataset='bodilex' samples_path=experiments/clean-motionfix/bodilex/bs64_1000ts_clip77/steps_1000_bodilex_noise_999/ld_txt-2.5_ld_mot-2.5"