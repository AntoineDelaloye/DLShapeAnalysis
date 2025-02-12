from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path
import argparse
import yaml
import sys
import nibabel as nib
import numpy as np

from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.models import ImplicitNetSegPrior, ImplicitNetSeparateSegPrior, ImplicitNetMountedSegPrior, \
    ImplicitNetSeparateSegLatent, ImplicitNetSegLatent, ImplicitNetMountedSegLatent
from data_loading.data_loader import Seg4DWholeImage_SAX, Seg3DWholeImage_SAX, Seg4DWholeImage_SAX_test, Seg4DWholeImage_SAX_UKB, Seg4DWholeImage_SAX_UKB_test
from utils import ValProgressBar

LATEST_CHECKPOINT_DIR = "latest_checkpoint"
BEST_WEIGHTS_PATH = "best_weights.pt"
CONFIG_SAVE_PATH = "config_cluster.yaml"


@dataclass
class Params:
    initial_val: bool = False
    pos_encoding: str = "none"  # nerf, none, gaussian
    num_frequencies: Tuple[int, ...] = (4, 4, 4)
    # num_frequencies: Tuple[int, ...] = (128,)
    freq_scale: float = 1.0
    latent_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.00
    num_hidden_layers: int = 8
    side_length: Tuple[int, ...] = (100, 100, -1, 1)
    coord_noise_std: float = 1e-4
    heart_pad: int = 10
    max_epochs: int = 3001
    log_interval: int = 1
    val_interval: int = 10
    fine_tune_max_epochs: int = 2001  # Lightning is faster doing (2000 epochs x 1 batch) than (1 epoch x 2000 batches)
    fine_tune_optimal_epochs: int = -1  # To be set during training
    fine_tune_log_interval: int = 500
    x_holdout_rate: int = 1  # Height
    y_holdout_rate: int = 1  # Width
    z_holdout_rate: int = 1  # Slices
    t_holdout_rate: int = 1  # Time
    rec_loss_weight: float = 1.0
    seg_loss_weight: float = 1.0
    seg_class_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    lr: float = 1e-4
    fine_tune_lr: float = 1e-4
    latent_reg: float = 1e-4
    weight_reg: float = 1e-4
    activation: str = "wire"  # periodic, relu, wire
    wire_omega_0: float = 10.  # WIRE Frequency term
    wire_scale_0: float = 10.  # WIRE Scaling term
    skip_connections: bool = True
    input_coord_to_all_layers: bool = False
    model_type: str = "shared"  # shared, separate, mounted
    augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    # augmentations: Tuple[Tuple[str, Dict[str, Any]], ...] = ({"translation", {"x_lim": 0.25, "y_lim": 0.25}},
    #                                                          {"gamma", {"gamma_lim": (0.7, 1.4)}})
    resume_checkpoint_path: str = None # Path from the log_dir if resuming training from previous trained weights


def init_model(dataset, val_dataset, **params):
    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegPrior(
            dataset=dataset,
            val_dataset=val_dataset,
            aug_num_parameters=dataset.num_aug_params,
            **params)
    else:
        raise NotImplementedError
    return model


def main_train(config_path: Optional[str] = None, exp_name: Optional[str] = None):
    # Config and hyper params
    config = {"params": {}}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    params = Params(**config["params"])
    root_dir = Path(config["log_dir"])

    # Dataset
    dataset = Seg4DWholeImage_SAX_UKB(load_dir=config["train_data_dir"],
                                  case_start_idx=config.get("train_start_idx", 0),
                                  num_cases=config["num_train"],
                                  **params.__dict__)
    coord_dimensions = dataset.sample_coords.shape[-1]
    assert coord_dimensions == len(params.side_length)
    train_dataloader = DataLoader(dataset, shuffle=True)

    val_dataset = Seg4DWholeImage_SAX_UKB_test(load_dir=config["val_data_dir"],
                                           case_start_idx=config.get("val_start_idx", config["num_train"]),
                                           num_cases=config["num_val"],
                                           **params.__dict__)
    # Model dir creation
    if exp_name is not None:
        root_dir = root_dir / exp_name
    root_dir = root_dir / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{str(dataset.__class__.__name__)}'
    os.makedirs(str(root_dir), exist_ok=True)
    print(root_dir)

    # Save config to model dir
    config["params"] = {k: v if not isinstance(v, tuple) else list(v) for k, v in params.__dict__.items()}
    with open(str(root_dir / CONFIG_SAVE_PATH), "w") as f:
        yaml.dump(config, f)

    # Model
    best_weights_path = root_dir / BEST_WEIGHTS_PATH
    model = init_model(dataset, val_dataset, best_checkpoint_path=best_weights_path, **params.__dict__)
    ckpt_latest_saver = ModelCheckpoint(save_top_k=1, dirpath=root_dir / LATEST_CHECKPOINT_DIR,
                                        monitor="step", mode="max")
    # Trainer
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=params.max_epochs, accelerator=accel,
                         default_root_dir=root_dir, callbacks=[ckpt_latest_saver])
    start = datetime.now()
    if params.resume_checkpoint_path is not None:
        trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=(Path(config["log_dir"]) / Path(params.resume_checkpoint_path)))
    else:
        trainer.fit(model, train_dataloaders=train_dataloader)
    print("Train elapsed time:", datetime.now() - start)

    # Save updated config to model dir
    params.fine_tune_optimal_epochs = model.overall_best_num_fine_tune_epochs
    config["params"] = {k: v if not isinstance(v, tuple) else list(v) for k, v in params.__dict__.items()}
    with open(str(root_dir / CONFIG_SAVE_PATH), "w") as f:
        yaml.dump(config, f)
    return best_weights_path


def main_eval(weights_path: str, config_path: Optional[str] = None):

    source_dir = Path(weights_path).parent

    # Load original model's config
    source_config_path = source_dir / CONFIG_SAVE_PATH
    source_config = {"params": {}}
    if source_config_path.exists():
        with open(str(source_config_path), "r") as f:
            source_config = yaml.safe_load(f)

    # Load user defined config
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Merged user defined config with original model config
    merged_params = {**source_config["params"], **config["params"]}  # User defined config takes precedence
    params = Params(**merged_params)
    params = params.__dict__

    if "log_dir" in config:
        # If user defined a log_dir use that one
        log_dir = config["log_dir"]
    else:
        # Otherwise defined log_dir beside original model's dir
        log_dir = str(source_dir.parent)
    root_dir = Path(log_dir) / (str(source_dir.name) + f'_test_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(str(root_dir), exist_ok=True)
    print(root_dir)

    # Define dataset and model
    dataset = Seg4DWholeImage_SAX_UKB_test(load_dir=config["test_data_dir"],
                                       case_start_idx=config.get("test_start_idx", config["num_train"] + config["num_val"]),
                                       num_cases=config["num_test"],
                                       **params)
    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegLatent(dataset=dataset, split_name="test", **params)
    else:
        raise ValueError("Unknown model type.")

    # Load trained model's weights
    sd = torch.load(weights_path)
    del sd["h"]
    a = model.load_state_dict(sd, strict=False)
    assert len(a.missing_keys) == 1 and a.missing_keys[0] == 'h'
    assert len(a.unexpected_keys) == 0
    # Fine tune model
    if params["fine_tune_optimal_epochs"] > 0:
        max_epochs = params["fine_tune_optimal_epochs"]
    else:
        max_epochs = params["fine_tune_max_epochs"]
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # accel = "gpu" if torch.cuda.is_available() else "cpu"
    accel = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator=accel,
                         default_root_dir=root_dir,
                         enable_model_summary=False,
                         enable_checkpointing=False, callbacks=[ValProgressBar()])
    trainer.fit(model, train_dataloaders=DataLoader(dataset, shuffle=False))

    # Save the model somewhere
    os.makedirs(source_dir, exist_ok=True)
    save_eval_path = os.path.join(source_dir, "evaluated_model.pt")
    try:        
        torch.save(model.state_dict(), save_eval_path)
    except:
        print("Error saving model")
        # Saving it in the data directory
        print("Saving it in the data directory")
        save_path = os.path.join(config["test_data_dir"], "derivatives", "DL_model")
        print(save_path)
        os.makedirs(save_path, exist_ok=True)
        save_eval_path = os.path.join(save_path, "evaluated_model.pt")
        torch.save(model.state_dict(), save_eval_path)


# def only_eval(weights_path: str, config_path: Optional[str] = None):
def main_test(weights_path: str, config_path: str = None, res_factor_z: float = 1.0):

    config = {"params": {}}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    params = Params(**config["params"])
    weights_path = Path(config["log_dir"] + "/" + weights_path)
    params = params.__dict__

    # if params["model_type"] == "separate":
    #     model = ImplicitNetSeparateSegLatent.load_from_checkpoint(weights_path, **params)
    # elif params["model_type"] == "shared":
    #     model = ImplicitNetSegLatent.load_from_checkpoint(weights_path, **params)
    # elif params["model_type"] == "mounted":
    #     model = ImplicitNetMountedSegLatent.load_from_checkpoint(weights_path, **params)
    # else:
    #     raise ValueError("Unknown model type.")
    
    

    # sd = torch.load(weights_path)
    # print(np.shape(sd["state_dict"]["h"]))
    # patients_im, patients_seg,_ = find_SAX_images_test(config["test_data_dir"], patients)
    data_dir_path = config["test_data_dir"]
    dataset = Seg4DWholeImage_SAX_UKB_test(load_dir=config["test_data_dir"],
                                       case_start_idx=config.get("test_start_idx", config["num_train"] + config["num_val"]),
                                       num_cases=config["num_test"],
                                       **params)
    
    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegLatent(dataset=dataset, split_name="test", **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegLatent(dataset=dataset, split_name="test", **params)
    else:
        raise ValueError("Unknown model type.")
    
    # patients_im, patients_seg, _ = dataset.find_images()
    # print("TEEEEEST", patients_im)
    model.eval()
    # print(np.shape(patients_seg))
    # localpatients_list = os.listdir("/home/ajdelalo/projects/DLShapeAnalysis/UKB_Dataset")
    # localpatients_list = sorted([i for i in localpatients_list if i.startswith("sub-")])

    # index_list = [1,4,8]
    # index_list = [1,2,3,4]
    for i in dataset.patients:
        # index_patient = np.squeeze([j for j in range(len(patients_list)) if patients_list[j] == localpatients_list[i]])
        # print(index_patient)
        if params["model_type"] == "separate":
            reconstruction, segmentation = ImplicitNetSeparateSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=i, res_factors=(1,1,res_factor_z))
        elif params["model_type"] == "shared":
            reconstruction, segmentation = ImplicitNetSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=i, res_factors=(1,1,res_factor_z))
        elif params["model_type"] == "mounted":
            reconstruction, segmentation = ImplicitNetMountedSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=i, res_factors=(1,1,res_factor_z))
        else:
            raise ValueError("Unknown model type.")
        
        # Save folder for each subject
        patient_id = dataset.patients[i]
        save_path = os.path.join(data_dir_path, 'results_dl_shape_baseline', f"sub-{patient_id}")
        os.makedirs(save_path, exist_ok=True)

        # Update the affine transformation
        nii_img = nib.load(dataset.im_paths[i])
        pixdim_low_res = nii_img.header['pixdim'][1:4]  # Should work
        # pixdim_low_res = np.linalg.norm(nii_img.affine[:3, :3], axis=0)  # Alternative
        original_shape = nii_img.get_fdata().shape
        new_shape = reconstruction.shape
        
        ratio_res = (np.asarray(new_shape) / np.asarray(original_shape))[:3]
        pixdim_high_res = pixdim_low_res / ratio_res
        
        # Create the new affine transformation
        new_affine = np.eye(4)
        new_affine[:3, :3] = np.diag(pixdim_high_res)

        nifti_seg = nib.Nifti1Image(segmentation, new_affine)
        # nib.save(nifti_seg, f"./results3/segmentation_{dataset.patients[i]}.nii.gz")
        nib.save(nifti_seg, os.path.join(save_path, f"sub-{patient_id}_seg.nii.gz"))

        nifti_image = nib.Nifti1Image(reconstruction, new_affine)
        nib.save(nifti_image, os.path.join(save_path, f"sub-{patient_id}_rec.nii.gz"))
        # nib.save(nifti_image, f"./results3/reconstruction_{dataset.patients[i]}.nii.gz")

    # Save results
    # print(np.shape(reconstruction))
    
    # print(np.shape(segmentation))
    # nifti_seg = nib.Nifti1Image(segmentation, np.eye(4))
    # nib.save(nifti_seg, "segmentation.nii.gz")

    # nifti_image = nib.Nifti1Image(reconstruction, np.eye(4))
    # nib.save(nifti_image, "reconstruction.nii.gz")


def parse_command_line():
    main_parser = argparse.ArgumentParser(description="Implicit Segmentation",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_subparsers = main_parser.add_subparsers(dest='pipeline')
    # train
    parser_train = main_subparsers.add_parser("train")
    parser_train.add_argument("-c", "--config",
                              help="path to configuration file", required=False,
                              default=r"/home/ajdelalo/projects/DLShapeAnalysis/configs/config_cluster.yaml"
                              )
    parser_train.add_argument("-n", "--exp_name",
                              help="custom experiment name", required=False,
                              default=""
                              )
    # eval
    parser_eval = main_subparsers.add_parser("eval")
    parser_eval.add_argument("-c", "--config",
                             help="path to configuration yml file", required=False,
                             )
    parser_eval.add_argument("-w", "--weights",
                             help="path to the desired checkpoint .ckpt file meant for evaluation", required=True,
                             )

    # Test only
    parser_test = main_subparsers.add_parser("test")
    parser_test.add_argument("-c", "--config",
                                help="path to configuration yml file", required=False,
                                default=r"/home/ajdelalo/projects/DLShapeAnalysis/configs/config_UKB.yaml"
                                )
    parser_test.add_argument("-w", "--weights",
                                help="path to the desired checkpoint .ckpt file meant for testing", required=False,
                                default="20241128-083257_Seg4DWholeImage_SAX_UKB/latest_checkpoint/epoch=808-step=404500.ckpt"
                                )
    parser_test.add_argument("-r", "--res_factor_z", default=1.0, type=float, help="Resolution factor for the z axis. Default is 1.0.")

    
    return main_parser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    if args.pipeline is None:
        sys.argv.append("test")
        args = parse_command_line()
        config_path, weights = args.config, args.weights
        main_test(weights, config_path)
    elif args.pipeline == "train":
        config_path, exp_name = args.config, args.exp_name
        weights_path, fine_tune_epochs = main_train(config_path, exp_name)
        main_eval(weights_path, config_path)
    elif args.pipeline == "eval":
        config_path, weights_path = args.config, args.weights
        main_eval(weights_path, config_path)
    elif args.pipeline == "test":
        weights_path, config_path, res_factor_z = args.weights, args.config, args.res_factor_z
        main_test(weights_path, config_path, res_factor_z)
    else:
        raise ValueError("Unknown pipeline selected.")
