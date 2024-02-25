from omegaconf import DictConfig
import logging
import hydra

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="text_motion_sim")
def text_motion_sim(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    npy_path = cfg.npy
    text = cfg.text

    import src.prepare  # noqa
    import torch
    import numpy as np
    from src.config import read_config
    from src.load import load_model_from_cfg
    from hydra.utils import instantiate
    from pytorch_lightning import seed_everything
    from src.data.collate import collate_x_dict
    from src.model.tmr import get_score_matrix

    cfg = read_config(run_dir)

    seed_everything(cfg.seed)

    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    motion = normalizer(motion)
    motion = motion.to(device)

    motion_x_dict = {"x": motion, "length": len(motion)}

    from src.data.text import SentenceEmbeddings
    from src.model.tmr import get_sim_matrix

    sent_embeds = SentenceEmbeddings(path='datasets/text_embeds/sentence/',
                       modelname='sentence-transformers/all-mpnet-base-v2',
                       preload=True)
    latents_text = []
    latents_motion = []

    with torch.inference_mode():
        for motion_x_dict, text in zip(motion_x_dict, text):
            # motion -> latent
            motion_x_dict = collate_x_dict([motion_x_dict])
            lat_m = model.encode(motion_x_dict, sample_mean=True)[0]

            # text -> latent
            text_x_dict = collate_x_dict(text_model([text]))
            lat_t = model.encode(text_x_dict, sample_mean=True)[0]

            latents_text.append(lat_t)
            latents_motion.append(lat_m)

            score = get_score_matrix(lat_t, lat_m).cpu()

    latent_texts = torch.cat(latents_text)
    latent_motions = torch.cat(latents_motion)
    sim_matrix = get_sim_matrix(latent_texts, latent_motions)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        # "sent_emb": sent_embs.cpu().numpy(),
    }

    # sent_embs = torch.cat(sent_embs)

    score_str = f"{score:.3}"
    logger.info(
        f"The similariy score s (0 <= s <= 1) between the text and the motion is: {score_str}"
    )


def compute_simitlarities(lotexts, lomotions):
    text_motion_sim(list_of_texts, list_of_motions)


if __name__ == "__main__":
    text_motion_sim()
