from rich import print
import time
import subprocess
import argparse
import os

def run(cmd):
    # print(f"---------Executing-----------\n {' '.join(cmd)}")
    # import ipdb; ipdb.set_trace()

    x = subprocess.run(cmd)

def main_loop(command, exp_paths, data):

    cmd_no=0
    cmd_eval = command
    from itertools import product
    paths_to_eval = exp_paths
    print("Number of different experiments is:", len(paths_to_eval))
    print('---------------------------------------------------')

    for fd in tqdm(paths_to_eval):
        cur_cmd = list(cmd_eval)
        idx_of_exp = cur_cmd.index("samples_path=FOLDER")
        cur_cmd[idx_of_exp] = f"samples_path={str(fd)} dataset={data}"
        # list_of_args = ' '.join([f'data={data}"'])
        # cur_cmd.extend([list_of_args])
        # `print(cur_cmd)
        run(cur_cmd)
        time.sleep(0.2)
        cmd_no += 1

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
    dataset = args.ds
    extras_args = args.extras
    start_index = extras_args.find("samples_path=") + len("samples_path=")
    end_index = extras_args.find(" ", start_index)
    if end_index == -1:  # In case there is no space after the target substring
        base_dir_of_samples = extras_args[start_index:]
    else:
        base_dir_of_samples = extras_args[start_index:end_index]
    samples_dirs_list = [ f.path for f in os.scandir(base_dir_of_samples) if f.is_dir() ]
    #samples_dirs_list = [os.path.join(base_dir_of_samples, d) for d in os.scandir(base_dir_of_samples)
    #                     if d.is_dir()]
    extras_args = extras_args.replace(base_dir_of_samples, 'FOLDER')
    print('---------------------------------------------------')
    assert dataset in ['bodilex', 'sinc_synth', 'hml3d']
    parser = argparse.ArgumentParser()
    cmd_eval = ['python', 'evaluate_cluster.py',
                 '--bid', '20',
                 '--extras', extras_args]
    
    main_loop(cmd_eval, samples_dirs_list,
              dataset)
    # python evaluate_cluster.py --extras "run_dir=outputs/tmr_humanml3d_amass_feats dataset='bodilex' samples_path=experiments/clean-motionfix/bodilex/bs64_1000ts_clip77/steps_1000_bodilex_noise_999/ld_txt-2.5_ld_mot-2.5"
