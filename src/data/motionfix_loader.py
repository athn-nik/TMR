import os
import torch
import numpy as np
import hydra
from src.data.smpl_fast import smpl_forward_fast
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple
from src.data.collate import collate_batch_last_padding
from pathlib import Path
import smplx
import joblib
from src.data.utils import cast_dict_to_tensors, freeze
from src.data.transforms3d import transform_body_pose, canonicalize_rotations
from einops import rearrange

class MotionFixLoader(Dataset):
# 'src.data.bodilex.BodilexDataModule',
#  'datapath = data/amass_bodilex_v5.pth.tar',
#  'smplh_path': 'data/body_models',
#  'load_splits': ['train', 'val', 'test'],
#   'preproc': {'stats_file': '/home/nathanasiou/Desktop/conditional_action_gen/modilex/deps/stats/statistics_bodilex.npy'
#               ,
#             'load_feats': ['body_transl_delta_pelv', 'body_orient', 'body_pose'],
    def __init__(self,
                 datapath: str = "",
                 smplh_path: str = "",
                 rot_repr: str = "6d",
                 **kwargs):
        # v11 is the next one 
        self.datapath = 'datasets/bodilex/amass_bodilex_v11.pth.tar'
        self.collate_fn = lambda b: collate_batch_last_padding(b, load_feats)
        self.rot_repr = rot_repr
        curdir = Path(hydra.utils.get_original_cwd())
        self.smpl_p = Path(curdir / 'datasets/body_models')
        # calculate splits
        self.normalizer = Normalizer(curdir/'stats/humanml3d/amass_feats')

        self.body_model = smplx.SMPLHLayer(f'{self.smpl_p}/smplh',
                                           model_type='smplh',
                                           gender='neutral',
                                           ext='npz').eval();
        setattr(smplx.SMPLHLayer, 'smpl_forward_fast', smpl_forward_fast)
        freeze(self.body_model)
        ds_db_path = Path(curdir / self.datapath)

        dataset_dict_raw = joblib.load(ds_db_path)
        dataset_dict_raw = cast_dict_to_tensors(dataset_dict_raw)
        for k, v in dataset_dict_raw.items():
            
            if len(v['motion_source']['rots'].shape) > 2:
                rots_flat_src = v['motion_source']['rots'].flatten(-2).float()
                dataset_dict_raw[k]['motion_source']['rots'] = rots_flat_src
            if len(v['motion_target']['rots'].shape) > 2:
                rots_flat_tgt = v['motion_target']['rots'].flatten(-2).float()
                dataset_dict_raw[k]['motion_target']['rots'] = rots_flat_tgt

            for mtype in ['motion_source', 'motion_target']:
            
                rots_can, trans_can = self._canonica_facefront(v[mtype]['rots'],
                                                               v[mtype]['trans']
                                                               )
                dataset_dict_raw[k][mtype]['rots'] = rots_can
                dataset_dict_raw[k][mtype]['trans'] = trans_can
                seqlen, jts_no = rots_can.shape[:2]
                
                rots_can_rotm = transform_body_pose(rots_can,
                                                  'aa->rot')
                # self.body_model.batch_size = seqlen * jts_no

                jts_can_ds = self.body_model.smpl_forward_fast(transl=trans_can,
                                                 body_pose=rots_can_rotm[:, 1:],
                                             global_orient=rots_can_rotm[:, :1])

                jts_can = jts_can_ds.joints[:, :22]
                dataset_dict_raw[k][mtype]['joint_positions'] = jts_can

        data_dict = cast_dict_to_tensors(dataset_dict_raw)
        data_ids = list(data_dict.keys())
        from src.data.utils import read_json
        splits = read_json(f'{os.path.dirname(Path(curdir / self.datapath))}/splits.json')
        test_ids = splits['test']
        self.motions = {}
        for test_id in test_ids:
            self.motions[test_id] = data_dict[test_id]
        self.keyids = list(self.motions.keys())

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)
    
    def _canonica_facefront(self, rotations, translation):
        rots_motion = rotations
        trans_motion = translation
        datum_len = rotations.shape[0]
        rots_motion_rotmat = transform_body_pose(rots_motion.reshape(datum_len,
                                                           -1, 3),
                                                           'aa->rot')
        orient_R_can, trans_can = canonicalize_rotations(rots_motion_rotmat[:,
                                                                             0],
                                                         trans_motion)            
        rots_motion_rotmat_can = rots_motion_rotmat
        rots_motion_rotmat_can[:, 0] = orient_R_can
        translation_can = trans_can - trans_can[0]
        rots_motion_aa_can = transform_body_pose(rots_motion_rotmat_can,
                                                 'rot->aa')
        rots_motion_aa_can = rearrange(rots_motion_aa_can, 'F J d -> F (J d)',
                                       d=3)
        return rots_motion_aa_can, translation_can

    def load_keyid(self, keyid):
        from prepare.compute_amass import _get_body_orient, _get_body_pose, _get_body_transl_delta_pelv
        source_m = self.motions[keyid]['motion_source']
        target_m = self.motions[keyid]['motion_target']
        text = self.motions[keyid]['text']
        # Take the first one for testing/validation
        # Otherwise take a random one        
        pose6d_src = _get_body_pose(source_m['rots'])
        orient6d_src = _get_body_orient(source_m['rots'][..., :3])
        trans_delta_src = _get_body_transl_delta_pelv(orient6d_src,
                                                  source_m['trans'])
        features_source = torch.cat([trans_delta_src, pose6d_src,
                                     orient6d_src], dim=-1)

        pose6d_tgt = _get_body_pose(target_m['rots'])
        orient6d_tgt = _get_body_orient(target_m['rots'][..., :3])
        trans_delta_tgt = _get_body_transl_delta_pelv(orient6d_tgt,
                                                  target_m['trans'])
        features_target = torch.cat([trans_delta_tgt, pose6d_tgt,
                                     orient6d_tgt], dim=-1)
        if self.normalizer is not None:
            features_source = self.normalizer(features_source)
            features_target = self.normalizer(features_target)

        output = {
            "motion_source": features_source,
            "motion_target": features_target,
            "text": text,
            "keyid": keyid,
        }
        return output

    # def __call__(self, path):
    #     # check if motion path exists
    #     if not os.path.exists(os.path.join(self.base_dir, path + ".npy")):
    #         self.not_found += 1
    #     motion_path = os.path.join(self.base_dir, path + ".npy")

    #     if path not in self.motions:
    #         motion = np.load(motion_path)
    #         motion = torch.from_numpy(motion).to(torch.float)
    #         if self.normalizer is not None:
    #             motion = self.normalizer(motion)
    #         self.motions[path] = motion
    #     motion = self.motions[path]

    #     x_dict = {"x": motion, "length": len(motion)}
    #     return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean, self.mean_path)
        torch.save(std, self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x
