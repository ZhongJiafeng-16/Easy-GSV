import os
import torch
from loguru import logger
from collections import OrderedDict

def export_lightning_ckpt_to_standard(path, exp_name):
    checkpoint = torch.load(path)
    
    new_state_dict = OrderedDict()
    new_state_dict['weight'] = checkpoint['state_dict']
    new_state_dict['config'] = checkpoint['hyper_parameters']['config']

    epoch = checkpoint["epoch"]
    step = checkpoint["global_step"]
    logger.info(f"Epoch: {epoch}, Step: {step}")
    save_path = os.path.join(os.path.dirname(path),f"{exp_name}-epoch={epoch}-step={step}.ckpt")
    torch.save(new_state_dict, save_path)
    logger.info(f"<{path}> has been exported to standard ckpt format at <{save_path}>")

if __name__ == '__main__':
    export_lightning_ckpt_to_standard("data/EN_B00005/ckpt/19-0.17.ckpt", "EN_B00005")