import os
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from .amass_utils import change_for, to_tensor, _canonica_facefront, transform_body_pose, flip_motion
from .amass_utils import fname_normalizer, path_normalizer, read_json, cast_dict_to_tensors
import numpy as np
import joblib
import smplx
import torch
from tqdm import tqdm
logger = logging.getLogger(__name__)


def extract_h3d(feats):
    from einops import unpack

    root_data, ric_data, rot_data, local_vel, feet_l, feet_r = unpack(
        feats, [[4], [63], [126], [66], [2], [2]], "i *"
    )
    return root_data, ric_data, rot_data, local_vel, feet_l, feet_r

def _get_body_pose(data):
    """get body pose"""
    # default is axis-angle representation: Frames x (Jx3) (J=21)
    pose = to_tensor(data[..., 3:3 + 21*3])  # drop pelvis orientation
    pose = transform_body_pose(pose, f"aa->6d")
    return pose

def _get_body_transl_delta_pelv(pelvis_orient, trans):
    """
    get body pelvis tranlation delta relative to pelvis coord.frame
    v_i = t_i - t_{i-1} relative to R_{i-1}
    """
    trans = to_tensor(trans)
    trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
    pelvis_orient = transform_body_pose(to_tensor(pelvis_orient), "6d->rot")
    trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
    trans_vel_pelv[0] = 0  # zero out velocity of first frame
    return trans_vel_pelv

def _get_body_transl(trans):
    """
    get body pelvis tranlation delta relative to pelvis coord.frame
    v_i = t_i - t_{i-1} relative to R_{i-1}
    """
    trans = to_tensor(trans)
    return trans

def _get_body_orient(orient):
    """get body global orientation"""
    # default is axis-angle representation
    pelvis_orient = to_tensor(orient)
    pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
    return pelvis_orient

def compute_amass():
    base_folder='/is/cluster/fast/nathanasiou/data/tmr_data/hml3d_amass'
    output_folder='./datasets/motions/amass_feats'

    force_redo=False # true to recompute the features

    from src.guofeats import joints_to_guofeats

    output_folder_M = os.path.join(output_folder, "M")
    body_model = smplx.SMPLHLayer('/is/cluster/fast/nathanasiou/data/motion-editing-project/body_models/smplh',
                                   model_type='smplh',
                                   gender='neutral',
                                   ext='npz').eval();

    print("The processed motions will be stored in this folder:")
    print(output_folder)
    dataset_dict_raw = joblib.load('/is/cluster/fast/nathanasiou/data/motion-editing-project/amass_smplhn/amass.pth.tar')
    dataset_dict_raw = cast_dict_to_tensors(dataset_dict_raw)
    for k, v in dataset_dict_raw.items():
        
        if len(v['rots'].shape) > 2:
            rots_flat_tgt = v['rots'].flatten(-2).float()
            dataset_dict_raw[k]['rots'] = rots_flat_tgt
        rots_can, trans_can = _canonica_facefront(v['rots'], v['trans'])
        dataset_dict_raw[k]['rots'] = rots_can
        dataset_dict_raw[k]['trans'] = trans_can
    path2key = read_json('/is/cluster/fast/nathanasiou/data/motion-editing-project/amass-mappings/amass_p2k.json')
    hml3d_annots = read_json('/is/cluster/fast/nathanasiou/data/motion-editing-project/annotations/humanml3d/annotations.json')
    for hml3d_key, key_annot in tqdm(hml3d_annots.items()):
        if 'humanact12/' in key_annot['path']:
            continue
        amass_norm_path = fname_normalizer(path_normalizer(key_annot['path']))
        cur_amass_key = path2key[amass_norm_path]
        if cur_amass_key not in dataset_dict_raw:
            continue
        text_and_durs = key_annot['annotations']
        cur_amass_data = dataset_dict_raw[cur_amass_key]
        dur_key = [sub_ann['end'] - sub_ann['start'] 
                    for sub_ann in text_and_durs]
        max_dur_id = dur_key.index(max(dur_key))
        text_annots = []
        for sub_ann in text_and_durs:
            if sub_ann['end'] - sub_ann['start'] <= 2:
                continue
            if sub_ann['end'] - sub_ann['start'] >= 10.1:
                continue
            text_annots.append(sub_ann['text'])
        if not text_annots:
            continue
        
        begin = int(text_and_durs[max_dur_id]['start'] * 30)
        end = int(text_and_durs[max_dur_id]['end'] * 30)
        rots_hml3d = cur_amass_data['rots'][begin:end]
        trans_hml3d = cur_amass_data['trans'][begin:end]

        if hml3d_key.startswith('M'):
            rots_mirr, trans_mirr = flip_motion(rots_hml3d,
                                                trans_hml3d)
            rots_mirr_rotm = transform_body_pose(rots_mirr,
                                                'aa->rot')
        
            jts_mirr_ds = body_model(transl=trans_mirr,
                                            body_pose=rots_mirr_rotm[:, 1:],
                                        global_orient=rots_mirr_rotm[:, :1])

            jts_can_mirr = jts_mirr_ds.joints[:, :22]
            jts_hml3d = jts_can_mirr
            rots_hml3d = rots_mirr
            trans_hml3d = trans_mirr
        else:
            jts_hml3d = cur_amass_data['joint_positions'][begin:end]

        pose6d = _get_body_pose(rots_hml3d)
        orient6d = _get_body_orient(rots_hml3d[..., :3])
        trans_delta = _get_body_transl_delta_pelv(orient6d, trans_hml3d)

        features = torch.cat([trans_delta, pose6d, orient6d], dim=-1)
        np.save(f'{output_folder}/{hml3d_key}.npy', features)


if __name__ == "__main__":
    compute_amass()
