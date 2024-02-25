import os
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)

mat2name = {
            'sim_matrix_s_t': 'source_target',
            'sim_matrix_t_t': 'target_generated'
            }

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)
from typing import List, Dict
import torch
from torch import Tensor

def lengths_to_mask_njoints(lengths: List[int], njoints: int, device: torch.device) -> Tensor:
    # joints*lenghts
    joints_lengths = [njoints*l for l in lengths]
    joints_mask = lengths_to_mask(joints_lengths, device)
    return joints_mask


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len,
                        device=device).expand(len(lengths),
                                              max_len) < lengths.unsqueeze(1)
    return mask

def collect_gen_samples(data_for_keys, motion_gen_path, normalizer, device):
    cur_samples = []
    # it becomes from 
    # translation | root_orient | rots --> trans | rots | root_orient 
    for i in range(len(data_for_keys)):
        cur_keyid = data_for_keys[i]['keyid']
        fname = f"{str(cur_keyid).zfill(6)}.npy"
        gen_motion_b = np.load(Path(motion_gen_path) / fname,
                                allow_pickle=True).item()['pose']
        gen_motion_b = torch.from_numpy(gen_motion_b)
        gen_motion_b_fixed = torch.zeros_like(gen_motion_b)
        gen_motion_b_fixed[:, -6:] = gen_motion_b[:, 3:9] 
        gen_motion_b_fixed[:, 3:-6] = gen_motion_b[:, 9:] 
        gen_motion_b_fixed[:, :3] = gen_motion_b[:, :3] 
        gen_motion_b_fixed = normalizer(gen_motion_b_fixed)
        cur_samples.append(gen_motion_b_fixed.to(device))
    return cur_samples

def compute_sim_matrix(model, dataset, keyids, motion_gen_path=None,
                       batch_size=256):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}

    if motion_gen_path is not None:
        curdir = Path(hydra.utils.get_original_cwd())
        # calculate splits
        from src.data.motionfix_loader import Normalizer
        normalizer = Normalizer(curdir/'stats/humanml3d/amass_feats')
        compute_gen = True
    else:
        compute_gen = False

    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        for sett in ['s_t', 't_t']:
            cur_samples = []
            latent_motions_A = []
            latent_motions_B = []
            for data in tqdm(all_data_splitted, leave=False):
                # batch = collate_text_motion(data, device=device)
                from src.data.collate import collate_tensor_with_padding, length_to_mask

                # TODO load the motions for the generations
                # Text is already encoded
                if sett == 's_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    if compute_gen:
                        cur_samples = collect_gen_samples(data, motion_gen_path,
                                                          normalizer, 
                                                          model.device)
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]


                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                                    'x': motion_b}
                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    if compute_gen:
                        cur_samples = collect_gen_samples(data, motion_gen_path,
                                                          normalizer, 
                                                          model.device)
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(cur_samples
                                                               ).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                    masks_a = length_to_mask(lengths_a, device=motion_a.device)
                    masks_b = length_to_mask(lengths_b, device=motion_b.device)
                    motion_a_dict = {'length': lengths_a, 'mask': masks_a,
                                    'x': motion_a}
                    motion_b_dict = {'length': lengths_b, 'mask': masks_b, 
                                    'x': motion_b}

                # Encode both motion and text
                latent_motion_A = model.encode(motion_a_dict, 
                                            sample_mean=True)
                latent_motion_B = model.encode(motion_b_dict,
                                            sample_mean=True)

                latent_motions_A.append(latent_motion_A)
                latent_motions_B.append(latent_motion_B)

            latent_motions_A = torch.cat(latent_motions_A)
            latent_motions_B = torch.cat(latent_motions_B)
            sim_matrix = get_sim_matrix(latent_motions_A, latent_motions_B)
            returned[f'sim_matrix_{sett}'] = sim_matrix.cpu().numpy()
    return returned

def get_motion_distances(model, dataset, keyids, motion_gen_path=None,
                         batch_size=256):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix
    import numpy as np
    device = model.device
    if batch_size > len(dataset):
        batch_size = len(dataset)
    nsplit = int(np.ceil(len(dataset) / batch_size))
    returned = {}
    import smplx
    body_model = smplx.SMPLHLayer(f'datasets/body_models/smplh',
                                    model_type='smplh',
                                    gender='neutral',
                                    ext='npz').to('cuda').eval();

    if motion_gen_path is not None:
        curdir = Path(hydra.utils.get_original_cwd())
        # calculate splits
        from src.data.motionfix_loader import Normalizer
        normalizer = Normalizer(curdir/'stats/humanml3d/amass_feats')
        compute_gen = True
    else:
        compute_gen = False

    with torch.no_grad():

        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        if nsplit > len(all_data):
            nsplit = len(all_data)
        all_data_splitted = np.array_split(all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        for sett in ['t_t']:
            cur_samples = []
            motions_a = []
            motions_b = []
            tot_lens_a = []
            tot_lens_b = []
            for data in tqdm(all_data_splitted, leave=False):
                # batch = collate_text_motion(data, device=device)
                from src.data.collate import collate_tensor_with_padding, length_to_mask
                # TODO load the motions for the generations
                # Text is already encoded
                if sett == 's_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_source'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_source']) for x in data]
                    if compute_gen:
                        cur_samples = collect_gen_samples(data, motion_gen_path,
                                                          normalizer, 
                                                          model.device)
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(
                            cur_samples).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding(
                           [x['motion_target'] for x in data]).to(model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                elif sett == 't_t':
                    motion_a = collate_tensor_with_padding(
                        [x['motion_target'] for x in data]).to(model.device)
                    lengths_a = [len(x['motion_target']) for x in data]
                    if compute_gen:
                        cur_samples = collect_gen_samples(data, motion_gen_path,
                                                          normalizer, 
                                                          model.device)
                        lengths_b = [len(x) for x in cur_samples]
                        motion_b = collate_tensor_with_padding(cur_samples
                                                               ).to(model.device)
                    else:
                        motion_b = collate_tensor_with_padding([
                            x['motion_target'] for x in data]).to(
                                model.device)
                        lengths_b = [len(x['motion_target']) for x in data]

                def split_into_chunks(N, k): 
                    chunked = [k*i for i in range(1, N//k+1)] + ([N] if N%k else [])
                    return [0] + chunked

                ids_for_smpl = split_into_chunks(motion_a.shape[0], 16)
                def sliding_window(lst):
                    return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]

                for s, e in sliding_window(ids_for_smpl):
                    motions_a.append(run_smpl_fwd(motion_a[s:e, :, :3],
                                                motion_a[s:e, :, -6:],
                                                motion_a[s:e, :, 3:-6],
                                                body_model))
                    motions_b.append(run_smpl_fwd(motion_b[s:e, :, :3],
                                                motion_b[s:e, :, -6:],
                                                motion_b[s:e, :, 3:-6],
                                                body_model))
                tot_lens_a.extend(lengths_a)
                tot_lens_b.extend(lengths_b)

            mask_a = lengths_to_mask(tot_lens_a, device).detach().cpu()
            mask_b = lengths_to_mask(tot_lens_b, device).detach().cpu()
            from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

            motions_a = torch.cat(motions_a).detach().cpu()
            motions_b = torch.cat(motions_b).detach().cpu()
            global_edit_accuracy = mse_loss(100*motions_a, 100*motions_b, 
                                           reduction='none').flatten(-2,-1).mean(-1)*mask_a
            tot_gl_edacc = global_edit_accuracy.sum() / mask_a.sum()

            # global_edit_accuracy = global_edit_accuracy.mean()

            returned[f'distances_{sett}'] = tot_gl_edacc.cpu().numpy()

    return returned

def run_smpl_fwd(body_transl, body_orient, body_pose, body_model,
                 verts=True):
    from src.data.transforms3d import transform_body_pose

    if len(body_transl.shape) > 2:
        bs, seqlen = body_transl.shape[:2]
        body_transl = body_transl.flatten(0, 1)
        body_orient = body_orient.flatten(0, 1)
        body_pose = body_pose.flatten(0, 1)

    batch_size = body_transl.shape[0]
    body_model.batch_size = batch_size
    verts = body_model(transl=body_transl, body_pose=transform_body_pose(body_pose,
                                                            '6d->rot'),
                      global_orient=transform_body_pose(body_orient,
                                                        '6d->rot')).vertices
    return verts.reshape(bs, seqlen, -1, 3)


@hydra.main(version_base=None, config_path="configs", 
            config_name="mot2mot_retrieval")
def retrieval(newcfg: DictConfig) -> None:
    protocol = newcfg.protocol
    threshold_val = newcfg.threshold
    device = newcfg.device
    run_dir = newcfg.run_dir
    ckpt_name = newcfg.ckpt
    batch_size = newcfg.batch_size
    motion_gen_path = newcfg.samples_path

    protocols = protocol

    save_dir = os.path.join(run_dir, "motionfix/contrastive_metrics")
    os.makedirs(save_dir, exist_ok=True)

    # Load last config
    from src.config import read_config
    import src.prepare  # noqa

    cfg = read_config(run_dir)

    import pytorch_lightning as pl
    import numpy as np
    from hydra.utils import instantiate
    from src.load import load_model_from_cfg
    from src.model.metrics import all_contrastive_metrics_mot2mot, print_latex_metrics_m2m

    pl.seed_everything(cfg.seed)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    datasets = {}
    results = {}
    bs_m2m = 32 # for the batch size metric

    for protocol in protocols:
        logger.info(f"|------Protocol {protocol.upper()}-----|")
        # Load the dataset if not already
        if protocol not in datasets:
            from src.data.motionfix_loader import MotionFixLoader
            from src.data.sincsynth_loader import SincSynthLoader
            if newcfg.dataset == 'sinc_synth':
                dataset = SincSynthLoader()
            else:
                dataset = MotionFixLoader()
            # TODO Load the motion editing test set
            
            datasets.update(
                {key: dataset for key in ["normal", "guo"]}
            )
        dataset = datasets[protocol]

        # Compute sim_matrix for each protocol
        if protocol not in results:
            if protocol=="normal":
                res = compute_sim_matrix(
                    model, dataset, dataset.keyids, 
                    motion_gen_path=motion_gen_path,
                    batch_size=batch_size,
                )
                results.update({key: res for key in ["normal"]})
                dists = get_motion_distances(
                    model, dataset, dataset.keyids, 
                    motion_gen_path=motion_gen_path,
                    batch_size=batch_size,
                )

            elif protocol == "guo":
                keyids = sorted(dataset.keyids)
                N = len(keyids)

                # make batches of 32
                idx = np.arange(N)
                np.random.seed(0)
                np.random.shuffle(idx)
                idx_batches = [
                    idx[bs_m2m * i : bs_m2m * (i + 1)] for i in range(len(keyids) // bs_m2m)
                ]

                # split into batches of 32
                # batched_keyids = [ [32], [32], [...]]
                results["guo"] = [
                    compute_sim_matrix(
                        model,
                        dataset,
                        np.array(keyids)[idx_batch],
                        motion_gen_path=motion_gen_path,
                        batch_size=batch_size,
                    )
                    for idx_batch in idx_batches
                ]
        result = results[protocol]

        # Compute the metrics
        if protocol == "guo":
            protocol_name = protocol
            def compute_guo_metrics(sim_matrix_lst):
                all_metrics = []
                for sim_matrix in sim_matrix_lst:
                    metrics = all_contrastive_metrics_mot2mot(sim_matrix,
                                                      rounding=None)
                    all_metrics.append(metrics)

                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = round(
                        float(np.mean([metrics[key] for metrics in all_metrics])), 2
                    )
                return avg_metrics
            metrics_dico = {}
            result_packed_to_d = {key: [d[key] for d in result]
                                  for key in result[0]
                                  }
            str_for_tab = ''
            for var, lst_of_sim_matrs in result_packed_to_d.items():
                logger.info(f'Case: {var} --> {mat2name[var]}')
                metr_name = mat2name[var]
                metrics_dico[metr_name] = compute_guo_metrics(lst_of_sim_matrs)
                str_for_tab += print_latex_metrics_m2m(metrics_dico[metr_name])
                metric_name = f"{protocol_name}_{metr_name}.yaml"
                path = os.path.join(save_dir, metric_name)
                save_metric(path, metrics_dico[metr_name])
                print(f"\n|-----------|\n")
            line_for_guo = str_for_tab.replace("\\\&", "&")
            print(f"\n|-----------||-----------||-----------||-----------|\n")

        else:
            protocol_name = protocol
            emb, threshold = None, None
            metrics = {}
            str_for_tab = ''
            for var, sim_matrix in result.items():
                logger.info(f'Case: {var} --> {mat2name[var]}')
                metr_name = mat2name[var]
                metrics[metr_name] = all_contrastive_metrics_mot2mot(sim_matrix, 
                                                emb, threshold=threshold)
                str_for_tab += print_latex_metrics_m2m(metrics[metr_name])

                metric_name = f"{protocol_name}_{metr_name}.yaml"
                path = os.path.join(save_dir, metric_name)
                save_metric(path, metrics[metr_name])
                print(f"\n|-----------|\n")
            line_for_all = str_for_tab.replace("\\\&", "&")
            print(f"\n|-----------||-----------||-----------||-----------|\n")
        if newcfg.samples_path is not None:
            short_expname = newcfg.samples_path.replace('/is/cluster/fast/nathanasiou/logs/motionfix-sigg/', '')
        else:
            short_expname = 'GroundTruth Results'
        logger.info(f"Testing done, metrics saved in:\n{path}")
        logger.info(f"-----------")

    print(f'----Experiment Folder----\n\n{short_expname}')
    print(f'----Batches of {bs_m2m}----\n\n{line_for_guo}')
    print(f'----Full Set----\n\n{line_for_all}')

if __name__ == "__main__":
    retrieval()