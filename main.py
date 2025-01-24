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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator=device,
                         default_root_dir=root_dir,
                         enable_model_summary=False,
                         enable_checkpointing=False, callbacks=[ValProgressBar()])
    trainer.fit(model, train_dataloaders=DataLoader(dataset, shuffle=False))

# def only_eval(weights_path: str, config_path: Optional[str] = None):
def main_test(weights_path: str, config_path: str = None):

    config = {"params": {}}
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    params = Params(**config["params"])
    weights_path = Path(config["log_dir"] + "/" + weights_path)
    params = params.__dict__

    if params["model_type"] == "separate":
        model = ImplicitNetSeparateSegLatent.load_from_checkpoint(weights_path, **params)
    elif params["model_type"] == "shared":
        model = ImplicitNetSegLatent.load_from_checkpoint(weights_path, **params)
    elif params["model_type"] == "mounted":
        model = ImplicitNetMountedSegLatent.load_from_checkpoint(weights_path, **params)
    else:
        raise ValueError("Unknown model type.")
    # sd = torch.load(weights_path)
    # print(np.shape(sd["state_dict"]["h"]))
    # patients_im, patients_seg,_ = find_SAX_images_test(config["test_data_dir"], patients)
    dataset = Seg4DWholeImage_SAX_UKB_test(load_dir=config["test_data_dir"],
                                       case_start_idx=config.get("test_start_idx", config["num_train"] + config["num_val"]),
                                       num_cases=config["num_test"],
                                       **params)
    # patients_im, patients_seg, _ = dataset.find_images()
    # print("TEEEEEST", patients_im)
    model.eval()
    # print(np.shape(patients_seg))
    patients_list = ['sub-1000372', 'sub-1004471', 'sub-1009925', 'sub-1012054', 'sub-1016533', 'sub-1016992', 'sub-1018052', 'sub-1022077', 'sub-1024367', 'sub-1026322', 'sub-1032720', 'sub-1033690', 'sub-1040227', 'sub-1042711', 'sub-1046939', 'sub-1051458', 'sub-1052171', 'sub-1053233', 'sub-1054329', 'sub-1055752', 'sub-1060967', 'sub-1063510', 'sub-1064732', 'sub-1067285', 'sub-1072476', 'sub-1072804', 'sub-1073856', 'sub-1074548', 'sub-1079915', 'sub-1081392', 'sub-1088475', 'sub-1091906', 'sub-1092665', 'sub-1092925', 'sub-1093377', 'sub-1101996', 'sub-1105124', 'sub-1106187', 'sub-1108059', 'sub-1108901', 'sub-1112328', 'sub-1115862', 'sub-1118773', 'sub-1123387', 'sub-1126834', 'sub-1129381', 'sub-1131961', 'sub-1133142', 'sub-1133612', 'sub-1137030', 'sub-1139189', 'sub-1142051', 'sub-1142387', 'sub-1143797', 'sub-1144220', 'sub-1148942', 'sub-1150414', 'sub-1153236', 'sub-1154053', 'sub-1154862', 'sub-1163690', 'sub-1165705', 'sub-1165804', 'sub-1165853', 'sub-1167063', 'sub-1169918', 'sub-1170788', 'sub-1174171', 'sub-1185504', 'sub-1188366', 'sub-1190057', 'sub-1204206', 'sub-1208387', 'sub-1209420', 'sub-1215621', 'sub-1217451', 'sub-1227864', 'sub-1229049', 'sub-1232835', 'sub-1234267', 'sub-1235861', 'sub-1236453', 'sub-1239821', 'sub-1244812', 'sub-1247861', 'sub-1248499', 'sub-1250337', 'sub-1254534', 'sub-1255945', 'sub-1256053', 'sub-1256332', 'sub-1258033', 'sub-1259056', 'sub-1260550', 'sub-1262695', 'sub-1264546', 'sub-1265786', 'sub-1268534', 'sub-1274235', 'sub-1276164', 'sub-1279948', 'sub-1281455', 'sub-1281747', 'sub-1282985', 'sub-1284622', 'sub-1285495', 'sub-1288087', 'sub-1292713', 'sub-1297310', 'sub-1298951', 'sub-1299633', 'sub-1300949', 'sub-1303245', 'sub-1304332', 'sub-1306954', 'sub-1307427', 'sub-1308295', 'sub-1313547', 'sub-1318623', 'sub-1319745', 'sub-1321113', 'sub-1325405', 'sub-1328845', 'sub-1333987', 'sub-1334962', 'sub-1335290', 'sub-1343402', 'sub-1343861', 'sub-1344020', 'sub-1344175', 'sub-1344296', 'sub-1347988', 'sub-1349564', 'sub-1352335', 'sub-1355820', 'sub-1356909', 'sub-1357656', 'sub-1362474', 'sub-1363087', 'sub-1366664', 'sub-1366696', 'sub-1368901', 'sub-1369776', 'sub-1371401', 'sub-1372359', 'sub-1383510', 'sub-1384834', 'sub-1387765', 'sub-1398179', 'sub-1405709', 'sub-1406203', 'sub-1411981', 'sub-1415320', 'sub-1415680', 'sub-1415854', 'sub-1416669', 'sub-1418048', 'sub-1420701', 'sub-1422145', 'sub-1422381', 'sub-1423382', 'sub-1423835', 'sub-1424589', 'sub-1425493', 'sub-1425702', 'sub-1427855', 'sub-1431836', 'sub-1433890', 'sub-1434148', 'sub-1438590', 'sub-1439598', 'sub-1441587', 'sub-1442121', 'sub-1446799', 'sub-1447976', 'sub-1450840', 'sub-1455418', 'sub-1459358', 'sub-1459387', 'sub-1460497', 'sub-1461535', 'sub-1469389', 'sub-1471388', 'sub-1471484', 'sub-1474397', 'sub-1474591', 'sub-1476390', 'sub-1479787', 'sub-1481916', 'sub-1481927', 'sub-1488335', 'sub-1491755', 'sub-1495580', 'sub-1500194', 'sub-1501274', 'sub-1503573', 'sub-1505614', 'sub-1510813', 'sub-1511384', 'sub-1514847', 'sub-1518113', 'sub-1518380', 'sub-1518475', 'sub-1520049', 'sub-1520666', 'sub-1521393', 'sub-1523229', 'sub-1525254', 'sub-1527839', 'sub-1528905', 'sub-1532078', 'sub-1539792', 'sub-1543834', 'sub-1550075', 'sub-1555068', 'sub-1567321', 'sub-1568152', 'sub-1571467', 'sub-1574950', 'sub-1580510', 'sub-1583220', 'sub-1588891', 'sub-1590160', 'sub-1599500', 'sub-1599823', 'sub-1599866', 'sub-1602703', 'sub-1603695', 'sub-1613271', 'sub-1614905', 'sub-1615767', 'sub-1615855', 'sub-1616133', 'sub-1616398', 'sub-1618211', 'sub-1619249', 'sub-1621010', 'sub-1626180', 'sub-1630289', 'sub-1637515', 'sub-1639645', 'sub-1642359', 'sub-1642568', 'sub-1645482', 'sub-1645596', 'sub-1651187', 'sub-1651509', 'sub-1651840', 'sub-1653770', 'sub-1674270', 'sub-1675938', 'sub-1688270', 'sub-1688463', 'sub-1690828', 'sub-1691034', 'sub-1695417', 'sub-1710216', 'sub-1711597', 'sub-1712459', 'sub-1712600', 'sub-1716719', 'sub-1720505', 'sub-1722218', 'sub-1728499', 'sub-1730783', 'sub-1741591', 'sub-1742150', 'sub-1742194', 'sub-1742308', 'sub-1744644', 'sub-1745327', 'sub-1751361', 'sub-1751574', 'sub-1758024', 'sub-1758725', 'sub-1758790', 'sub-1760369', 'sub-1762807', 'sub-1763545', 'sub-1763942', 'sub-1764337', 'sub-1764918', 'sub-1766054', 'sub-1767575', 'sub-1767837', 'sub-1767879', 'sub-1768713', 'sub-1774599', 'sub-1775070', 'sub-1778344', 'sub-1780906', 'sub-1781979', 'sub-1782574', 'sub-1783184', 'sub-1786391', 'sub-1788082', 'sub-1788808', 'sub-1790810', 'sub-1795702', 'sub-1796216', 'sub-1799443', 'sub-1799495', 'sub-1802991', 'sub-1805094', 'sub-1805938', 'sub-1809471', 'sub-1818816', 'sub-1820757', 'sub-1821522', 'sub-1822164', 'sub-1823871', 'sub-1843090', 'sub-1845155', 'sub-1848183', 'sub-1848884', 'sub-1852071', 'sub-1857939', 'sub-1860256', 'sub-1860557', 'sub-1866292', 'sub-1868310', 'sub-1878074', 'sub-1879552', 'sub-1880739', 'sub-1880857', 'sub-1881715', 'sub-1883841', 'sub-1884965', 'sub-1892977', 'sub-1905551', 'sub-1905846', 'sub-1906832', 'sub-1907491', 'sub-1908703', 'sub-1917135', 'sub-1920024', 'sub-1923415', 'sub-1928180', 'sub-1930291', 'sub-1931588', 'sub-1939267', 'sub-1939392', 'sub-1940006', 'sub-1940705', 'sub-1940827', 'sub-1942496', 'sub-1946157', 'sub-1947862', 'sub-1947944', 'sub-1948550', 'sub-1951528', 'sub-1963442', 'sub-1964368', 'sub-1970406', 'sub-1976578', 'sub-1979672', 'sub-1986004', 'sub-1987530', 'sub-1988088', 'sub-1995435', 'sub-1996043', 'sub-1997093', 'sub-2001006', 'sub-2010172', 'sub-2012055', 'sub-2012585', 'sub-2012756', 'sub-2017271', 'sub-2017574', 'sub-2018429', 'sub-2019369', 'sub-2022073', 'sub-2026558', 'sub-2029744', 'sub-2031462', 'sub-2033465', 'sub-2033746', 'sub-2036555', 'sub-2042742', 'sub-2043476', 'sub-2053873', 'sub-2056636', 'sub-2058248', 'sub-2058675', 'sub-2062143', 'sub-2063438', 'sub-2066722', 'sub-2067251', 'sub-2068557', 'sub-2069445', 'sub-2072161', 'sub-2077674', 'sub-2079234', 'sub-2083370', 'sub-2088578', 'sub-2089658', 'sub-2091133', 'sub-2109897', 'sub-2111099', 'sub-2127546', 'sub-2131665', 'sub-2133100', 'sub-2138626', 'sub-2139583', 'sub-2142413', 'sub-2144175', 'sub-2145882', 'sub-2146084', 'sub-2146334', 'sub-2149150', 'sub-2151168', 'sub-2156413', 'sub-2156574', 'sub-2157656', 'sub-2161858', 'sub-2166335', 'sub-2166490', 'sub-2172744', 'sub-2173712', 'sub-2180556', 'sub-2185914', 'sub-2188470', 'sub-2196663', 'sub-2199247', 'sub-2200465', 'sub-2200930', 'sub-2202665', 'sub-2203957', 'sub-2205197', 'sub-2208361', 'sub-2209457', 'sub-2213867', 'sub-2217035', 'sub-2217177', 'sub-2217226', 'sub-2220049', 'sub-2224742', 'sub-2225646', 'sub-2229098', 'sub-2230438', 'sub-2238920', 'sub-2246257', 'sub-2249800', 'sub-2250477', 'sub-2251275', 'sub-2251709', 'sub-2254095', 'sub-2256945', 'sub-2257797', 'sub-2262840', 'sub-2264987', 'sub-2270892', 'sub-2274179', 'sub-2278939', 'sub-2282717', 'sub-2287976', 'sub-2298080', 'sub-2298451', 'sub-2302490', 'sub-2303447', 'sub-2304231', 'sub-2304342', 'sub-2309289', 'sub-2311025', 'sub-2316987', 'sub-2319691', 'sub-2319930', 'sub-2320110', 'sub-2321101', 'sub-2321792', 'sub-2321910', 'sub-2324162', 'sub-2324441', 'sub-2324748', 'sub-2326248', 'sub-2326267', 'sub-2332048', 'sub-2335013', 'sub-2338539', 'sub-2340316', 'sub-2341512', 'sub-2343918', 'sub-2349855', 'sub-2350093', 'sub-2352079', 'sub-2353801', 'sub-2356895', 'sub-2366597', 'sub-2371821', 'sub-2376170', 'sub-2381480', 'sub-2385772', 'sub-2402459', 'sub-2403005', 'sub-2405026', 'sub-2407237', 'sub-2408314', 'sub-2411993', 'sub-2415441', 'sub-2421236', 'sub-2425030', 'sub-2426546', 'sub-2429965', 'sub-2430735', 'sub-2439090', 'sub-2439275', 'sub-2440945', 'sub-2442970', 'sub-2444551', 'sub-2449300', 'sub-2455920', 'sub-2456756', 'sub-2461148', 'sub-2465255', 'sub-2465373', 'sub-2465940', 'sub-2467535', 'sub-2470046', 'sub-2471914', 'sub-2472111', 'sub-2474976', 'sub-2478016', 'sub-2484041', 'sub-2485266', 'sub-2485303', 'sub-2485900', 'sub-2486338', 'sub-2488345', 'sub-2488393', 'sub-2493941', 'sub-2495706', 'sub-2497449', 'sub-2510547', 'sub-2514134', 'sub-2515057', 'sub-2515605', 'sub-2515917', 'sub-2519048', 'sub-2522601', 'sub-2525072', 'sub-2529662', 'sub-2530509', 'sub-2534129', 'sub-2534634', 'sub-2537241', 'sub-2537436', 'sub-2537928', 'sub-2538744', 'sub-2541379', 'sub-2545552', 'sub-2546111', 'sub-2546795', 'sub-2548544', 'sub-2549165', 'sub-2549486', 'sub-2549801', 'sub-2550636', 'sub-2569562', 'sub-2573798', 'sub-2575116', 'sub-2578617', 'sub-2584579', 'sub-2586929', 'sub-2592941', 'sub-2594553', 'sub-2597285', 'sub-2597567', 'sub-2598689', 'sub-2599560', 'sub-2600575', 'sub-2603279', 'sub-2605852', 'sub-2608418', 'sub-2611591', 'sub-2615252', 'sub-2622111', 'sub-2622346', 'sub-2623105', 'sub-2625335', 'sub-2629244', 'sub-2632580', 'sub-2632839', 'sub-2633067', 'sub-2637553', 'sub-2643021', 'sub-2643641', 'sub-2644930', 'sub-2648254', 'sub-2648341', 'sub-2660407', 'sub-2660727', 'sub-2661445', 'sub-2661673', 'sub-2665663', 'sub-2666878', 'sub-2668734', 'sub-2671248', 'sub-2672311', 'sub-2674015', 'sub-2681046', 'sub-2681628', 'sub-2683420', 'sub-2684031', 'sub-2685326', 'sub-2688204', 'sub-2692309', 'sub-2693195', 'sub-2696270', 'sub-2698476', 'sub-2698777', 'sub-2701169', 'sub-2701172', 'sub-2701864', 'sub-2702810', 'sub-2703558', 'sub-2707325', 'sub-2708026', 'sub-2708276', 'sub-2711298', 'sub-2713101', 'sub-2717627', 'sub-2718137', 'sub-2724065', 'sub-2725909', 'sub-2729339', 'sub-2734107', 'sub-2740413', 'sub-2743334', 'sub-2745756', 'sub-2747759', 'sub-2750212', 'sub-2751895', 'sub-2753387', 'sub-2755168', 'sub-2766860', 'sub-2766996', 'sub-2772073', 'sub-2773382', 'sub-2775895', 'sub-2780726', 'sub-2781996', 'sub-2786882', 'sub-2788410', 'sub-2790678', 'sub-2794633', 'sub-2799141', 'sub-2806411', 'sub-2807654', 'sub-2808156', 'sub-2817466', 'sub-2819836', 'sub-2820355', 'sub-2820452', 'sub-2827847', 'sub-2831938', 'sub-2835011', 'sub-2838625', 'sub-2840320', 'sub-2842932', 'sub-2852589', 'sub-2853364', 'sub-2856124', 'sub-2857351', 'sub-2857930', 'sub-2858411', 'sub-2862558', 'sub-2866342', 'sub-2867675', 'sub-2869200', 'sub-2871002', 'sub-2871786', 'sub-2872067', 'sub-2872157', 'sub-2876416', 'sub-2882916', 'sub-2884269', 'sub-2885826', 'sub-2886937', 'sub-2887065', 'sub-2892492', 'sub-2901837', 'sub-2906477', 'sub-2907189', 'sub-2908327', 'sub-2910743', 'sub-2915384', 'sub-2916263', 'sub-2918498', 'sub-2919770', 'sub-2921652', 'sub-2923811', 'sub-2930320', 'sub-2937133', 'sub-2937207', 'sub-2938909', 'sub-2939904', 'sub-2942106', 'sub-2942761', 'sub-2947857', 'sub-2954078', 'sub-2963618', 'sub-2968289', 'sub-2969387', 'sub-2969639', 'sub-2974180', 'sub-2975522', 'sub-2980635', 'sub-2982795', 'sub-2982885', 'sub-2983347', 'sub-2985047', 'sub-2985372', 'sub-2986857', 'sub-2987781', 'sub-2991557', 'sub-2995721', 'sub-2996444', 'sub-3000298', 'sub-3007871', 'sub-3007938', 'sub-3011567', 'sub-3012081', 'sub-3012470', 'sub-3025212', 'sub-3026053', 'sub-3026808', 'sub-3029774', 'sub-3031667', 'sub-3032066', 'sub-3038670', 'sub-3041928', 'sub-3046656', 'sub-3046985', 'sub-3049952', 'sub-3052465', 'sub-3055076', 'sub-3056165', 'sub-3068061', 'sub-3075073', 'sub-3076030', 'sub-3077443', 'sub-3078218', 'sub-3078376', 'sub-3085862', 'sub-3095296', 'sub-3096563', 'sub-3100719', 'sub-3101515', 'sub-3102251', 'sub-3106015', 'sub-3112049', 'sub-3112926', 'sub-3116771', 'sub-3117797', 'sub-3118230', 'sub-3121555', 'sub-3125943', 'sub-3128632', 'sub-3128940', 'sub-3137358', 'sub-3144111', 'sub-3146839', 'sub-3149553', 'sub-3151260', 'sub-3153797', 'sub-3162993', 'sub-3164376', 'sub-3165239', 'sub-3166345', 'sub-3166866', 'sub-3169995', 'sub-3172013', 'sub-3172760', 'sub-3175697', 'sub-3175929', 'sub-3180021', 'sub-3191552', 'sub-3198365', 'sub-3201468', 'sub-3202400', 'sub-3204124', 'sub-3206250', 'sub-3206483', 'sub-3209395', 'sub-3210026', 'sub-3210198', 'sub-3212380', 'sub-3215708', 'sub-3216393', 'sub-3219149', 'sub-3223220', 'sub-3225291', 'sub-3225331', 'sub-3231066', 'sub-3235985', 'sub-3237401', 'sub-3246963', 'sub-3249568', 'sub-3250133', 'sub-3255319', 'sub-3258960', 'sub-3263646', 'sub-3263860', 'sub-3267709', 'sub-3267742', 'sub-3271671', 'sub-3278626', 'sub-3282139', 'sub-3287481', 'sub-3288157', 'sub-3290768', 'sub-3291158', 'sub-3291381', 'sub-3295003', 'sub-3295979', 'sub-3296225', 'sub-3297679', 'sub-3297849', 'sub-3298385', 'sub-3299265', 'sub-3302288', 'sub-3302911', 'sub-3307693', 'sub-3308154', 'sub-3308587', 'sub-3317760', 'sub-3320204', 'sub-3327158', 'sub-3329796', 'sub-3333627', 'sub-3334676', 'sub-3335596', 'sub-3343968', 'sub-3345714', 'sub-3346500', 'sub-3350321', 'sub-3351897', 'sub-3353508', 'sub-3354768', 'sub-3354892', 'sub-3356870', 'sub-3363687', 'sub-3364077', 'sub-3368698', 'sub-3370110', 'sub-3370747', 'sub-3372680', 'sub-3375304', 'sub-3377326', 'sub-3385520', 'sub-3389342', 'sub-3390397', 'sub-3392543', 'sub-3392586', 'sub-3395441', 'sub-3399138', 'sub-3400071', 'sub-3401265', 'sub-3412281', 'sub-3412848', 'sub-3413673', 'sub-3417092', 'sub-3417782', 'sub-3418648', 'sub-3420114', 'sub-3420186', 'sub-3428900', 'sub-3430449', 'sub-3431848', 'sub-3434059', 'sub-3446207', 'sub-3452516', 'sub-3455408', 'sub-3460005', 'sub-3464373', 'sub-3467795', 'sub-3470164', 'sub-3471009', 'sub-3471797', 'sub-3473746', 'sub-3475322', 'sub-3475597', 'sub-3476591', 'sub-3477807', 'sub-3478247', 'sub-3481826', 'sub-3481973', 'sub-3484946', 'sub-3485030', 'sub-3490531', 'sub-3491347', 'sub-3492919', 'sub-3493007', 'sub-3493565', 'sub-3496556', 'sub-3498085', 'sub-3500518', 'sub-3502178', 'sub-3505903', 'sub-3507745', 'sub-3509459', 'sub-3510312', 'sub-3510579', 'sub-3514727', 'sub-3516176', 'sub-3518653', 'sub-3519042', 'sub-3526711', 'sub-3527980', 'sub-3532228', 'sub-3534010', 'sub-3534767', 'sub-3543683', 'sub-3548271', 'sub-3549105', 'sub-3553024', 'sub-3555282', 'sub-3556604', 'sub-3563654', 'sub-3564572', 'sub-3567225', 'sub-3567479', 'sub-3568284', 'sub-3568650', 'sub-3573381', 'sub-3577785', 'sub-3580615', 'sub-3581888', 'sub-3583978', 'sub-3586709', 'sub-3587552', 'sub-3592441', 'sub-3594003', 'sub-3594238', 'sub-3594530', 'sub-3594979', 'sub-3599354', 'sub-3600686', 'sub-3601303', 'sub-3603528', 'sub-3609119', 'sub-3609602', 'sub-3616226', 'sub-3616838', 'sub-3617251', 'sub-3618691', 'sub-3620238', 'sub-3620717', 'sub-3620943', 'sub-3626865', 'sub-3627595', 'sub-3630655', 'sub-3631321', 'sub-3631826', 'sub-3632528', 'sub-3633756', 'sub-3634300', 'sub-3634490', 'sub-3635144', 'sub-3640581', 'sub-3640976', 'sub-3644477', 'sub-3645003', 'sub-3648810', 'sub-3649558', 'sub-3651204', 'sub-3656106', 'sub-3656634', 'sub-3658644', 'sub-3659355', 'sub-3662457', 'sub-3664471', 'sub-3664966', 'sub-3667155', 'sub-3668366', 'sub-3668964', 'sub-3670366', 'sub-3678578', 'sub-3682743', 'sub-3691106', 'sub-3691320', 'sub-3691680', 'sub-3695997', 'sub-3699675', 'sub-3705856', 'sub-3716038', 'sub-3719961', 'sub-3723238', 'sub-3724499', 'sub-3724849', 'sub-3728859', 'sub-3730780', 'sub-3737690', 'sub-3738503', 'sub-3739504', 'sub-3744091', 'sub-3750621', 'sub-3751077', 'sub-3751896', 'sub-3755776', 'sub-3756693', 'sub-3759417', 'sub-3761731', 'sub-3762516', 'sub-3766177', 'sub-3766402', 'sub-3766852', 'sub-3771287', 'sub-3775250', 'sub-3775753', 'sub-3786892', 'sub-3791090', 'sub-3796525', 'sub-3796695', 'sub-3797550', 'sub-3800341', 'sub-3804524', 'sub-3807854', 'sub-3810087', 'sub-3811754', 'sub-3821499', 'sub-3828531', 'sub-3830743', 'sub-3831345', 'sub-3837649', 'sub-3842253', 'sub-3845983', 'sub-3846156', 'sub-3847756', 'sub-3847902', 'sub-3850488', 'sub-3850982', 'sub-3854059', 'sub-3854704', 'sub-3855178', 'sub-3859420', 'sub-3867579', 'sub-3867736', 'sub-3867852', 'sub-3870782', 'sub-3870945', 'sub-3871039', 'sub-3871205', 'sub-3874883', 'sub-3875752', 'sub-3879517', 'sub-3881467', 'sub-3883837', 'sub-3887273', 'sub-3887974', 'sub-3895242', 'sub-3898195', 'sub-3900489', 'sub-3902875', 'sub-3903708', 'sub-3904439', 'sub-3906429', 'sub-3907862', 'sub-3911641', 'sub-3911714', 'sub-3912709', 'sub-3918973', 'sub-3919651', 'sub-3930046', 'sub-3930258', 'sub-3932015', 'sub-3932271', 'sub-3932850', 'sub-3939213', 'sub-3941706', 'sub-3944238', 'sub-3944995', 'sub-3948738', 'sub-3949702', 'sub-3951647', 'sub-3957618', 'sub-3958566', 'sub-3959243', 'sub-3960478', 'sub-3962825', 'sub-3963053', 'sub-3963552', 'sub-3970217', 'sub-3981756', 'sub-3985233', 'sub-3992409', 'sub-3998829', 'sub-4000972', 'sub-4004735', 'sub-4011432', 'sub-4015406', 'sub-4016323', 'sub-4018118', 'sub-4019519', 'sub-4020748', 'sub-4022282', 'sub-4026771', 'sub-4029200', 'sub-4029918', 'sub-4033365', 'sub-4036720', 'sub-4040467', 'sub-4042013', 'sub-4044848', 'sub-4049925', 'sub-4054102', 'sub-4056551', 'sub-4059155', 'sub-4060452', 'sub-4061142', 'sub-4063322', 'sub-4064514', 'sub-4064787', 'sub-4068246', 'sub-4068986', 'sub-4069602', 'sub-4072631', 'sub-4077159', 'sub-4080710', 'sub-4081114', 'sub-4092172', 'sub-4094367', 'sub-4095659', 'sub-4098145', 'sub-4098839', 'sub-4100738', 'sub-4106721', 'sub-4107133', 'sub-4110365', 'sub-4112071', 'sub-4113521', 'sub-4114352', 'sub-4115110', 'sub-4116853', 'sub-4120049', 'sub-4123802', 'sub-4125902', 'sub-4126676', 'sub-4127555', 'sub-4133152', 'sub-4135001', 'sub-4135498', 'sub-4135617', 'sub-4138283', 'sub-4138577', 'sub-4139068', 'sub-4139670', 'sub-4144981', 'sub-4146358', 'sub-4147465', 'sub-4149742', 'sub-4150789', 'sub-4151412', 'sub-4160625', 'sub-4161854', 'sub-4162158', 'sub-4163518', 'sub-4165997', 'sub-4174891', 'sub-4179481', 'sub-4180575', 'sub-4182224', 'sub-4183448', 'sub-4184211', 'sub-4185632', 'sub-4186752', 'sub-4187215', 'sub-4188258', 'sub-4191438', 'sub-4193088', 'sub-4197004', 'sub-4197788', 'sub-4198203', 'sub-4209929', 'sub-4211783', 'sub-4212335', 'sub-4214294', 'sub-4215367', 'sub-4218387', 'sub-4230712', 'sub-4231397', 'sub-4234459', 'sub-4239399', 'sub-4242288', 'sub-4243486', 'sub-4246419', 'sub-4253863', 'sub-4261828', 'sub-4266355', 'sub-4268715', 'sub-4269106', 'sub-4280983', 'sub-4282722', 'sub-4288078', 'sub-4288203', 'sub-4293084', 'sub-4293452', 'sub-4296028', 'sub-4296415', 'sub-4296722', 'sub-4297975', 'sub-4299851', 'sub-4301102', 'sub-4309596', 'sub-4313985', 'sub-4317655', 'sub-4318853', 'sub-4321111', 'sub-4321472', 'sub-4322897', 'sub-4325736', 'sub-4328553', 'sub-4330381', 'sub-4338055', 'sub-4348110', 'sub-4350025', 'sub-4355322', 'sub-4356615', 'sub-4356777', 'sub-4357520', 'sub-4358023', 'sub-4363080', 'sub-4367523', 'sub-4373066', 'sub-4378783', 'sub-4379583', 'sub-4382414', 'sub-4382728', 'sub-4385521', 'sub-4388800', 'sub-4389467', 'sub-4391313', 'sub-4392351', 'sub-4392832', 'sub-4395412', 'sub-4400212', 'sub-4403455', 'sub-4405093', 'sub-4407088', 'sub-4410096', 'sub-4412681', 'sub-4415305', 'sub-4415462', 'sub-4423655', 'sub-4430106', 'sub-4441615', 'sub-4442680', 'sub-4449375', 'sub-4451926', 'sub-4452054', 'sub-4454547', 'sub-4455464', 'sub-4463978', 'sub-4467390', 'sub-4471953', 'sub-4473024', 'sub-4474071', 'sub-4474668', 'sub-4474673', 'sub-4475847', 'sub-4476152', 'sub-4482148', 'sub-4486833', 'sub-4487225', 'sub-4487289', 'sub-4487568', 'sub-4487610', 'sub-4490166', 'sub-4493835', 'sub-4501028', 'sub-4503656', 'sub-4511492', 'sub-4518096', 'sub-4523560', 'sub-4524351', 'sub-4525516', 'sub-4526102', 'sub-4528254', 'sub-4528624', 'sub-4529079', 'sub-4533355', 'sub-4534133', 'sub-4538281', 'sub-4539474', 'sub-4547305', 'sub-4549427', 'sub-4551502', 'sub-4551905', 'sub-4553658', 'sub-4554294', 'sub-4554637', 'sub-4557553', 'sub-4564522', 'sub-4564806', 'sub-4566454', 'sub-4568100', 'sub-4571348', 'sub-4575905', 'sub-4577934', 'sub-4579882', 'sub-4583895', 'sub-4584565', 'sub-4585149', 'sub-4585275', 'sub-4586594', 'sub-4589720', 'sub-4589831', 'sub-4595793', 'sub-4598010', 'sub-4600476', 'sub-4602184', 'sub-4602520', 'sub-4602965', 'sub-4603428', 'sub-4609455', 'sub-4610981', 'sub-4612339', 'sub-4613136', 'sub-4620610', 'sub-4633470', 'sub-4634948', 'sub-4635050', 'sub-4638354', 'sub-4639019', 'sub-4639832', 'sub-4640829', 'sub-4647220', 'sub-4648918', 'sub-4649928', 'sub-4650206', 'sub-4657472', 'sub-4659476', 'sub-4662533', 'sub-4667740', 'sub-4668464', 'sub-4669850', 'sub-4670701', 'sub-4674782', 'sub-4680734', 'sub-4688156', 'sub-4695802', 'sub-4699774', 'sub-4699953', 'sub-4700111', 'sub-4703958', 'sub-4706939', 'sub-4707279', 'sub-4707344', 'sub-4714022', 'sub-4718165', 'sub-4720977', 'sub-4721236', 'sub-4721767', 'sub-4722455', 'sub-4723756', 'sub-4724050', 'sub-4730140', 'sub-4735240', 'sub-4736175', 'sub-4737521', 'sub-4742624', 'sub-4745549', 'sub-4750750', 'sub-4751432', 'sub-4756568', 'sub-4756756', 'sub-4756970', 'sub-4765611', 'sub-4766471', 'sub-4767712', 'sub-4768732', 'sub-4774249', 'sub-4775160', 'sub-4775766', 'sub-4778332', 'sub-4786623', 'sub-4788183', 'sub-4788925', 'sub-4790800', 'sub-4790937', 'sub-4794692', 'sub-4794991', 'sub-4796517', 'sub-4798285', 'sub-4798424', 'sub-4799357', 'sub-4802730', 'sub-4802924', 'sub-4804317', 'sub-4805325', 'sub-4810789', 'sub-4812894', 'sub-4821580', 'sub-4821688', 'sub-4822281', 'sub-4824302', 'sub-4824613', 'sub-4826700', 'sub-4827128', 'sub-4828793', 'sub-4829428', 'sub-4832778', 'sub-4833448', 'sub-4833853', 'sub-4834964', 'sub-4836019', 'sub-4836269', 'sub-4837877', 'sub-4837892', 'sub-4839113', 'sub-4843876', 'sub-4849748', 'sub-4850395', 'sub-4852866', 'sub-4856702', 'sub-4860479', 'sub-4861470', 'sub-4862229', 'sub-4866785', 'sub-4866884', 'sub-4877422', 'sub-4881183', 'sub-4882222', 'sub-4882613', 'sub-4888854', 'sub-4890342', 'sub-4893841', 'sub-4895521', 'sub-4895596', 'sub-4900859', 'sub-4907391', 'sub-4920069', 'sub-4921129', 'sub-4923505', 'sub-4928478', 'sub-4930494', 'sub-4930813', 'sub-4931203', 'sub-4932223', 'sub-4934991', 'sub-4936237', 'sub-4936295', 'sub-4940016', 'sub-4940487', 'sub-4944495', 'sub-4946297', 'sub-4949023', 'sub-4954663', 'sub-4955571', 'sub-4956048', 'sub-4958305', 'sub-4958547', 'sub-4969510', 'sub-4972768', 'sub-4978310', 'sub-4979654', 'sub-4979899', 'sub-4983754', 'sub-4990855', 'sub-4993368', 'sub-4995369', 'sub-4999439', 'sub-5003225', 'sub-5007196', 'sub-5008593', 'sub-5009097', 'sub-5010835', 'sub-5011153', 'sub-5011594', 'sub-5017690', 'sub-5025165', 'sub-5026899', 'sub-5030023', 'sub-5031685', 'sub-5040009', 'sub-5049067', 'sub-5052894', 'sub-5053123', 'sub-5056206', 'sub-5057512', 'sub-5057964', 'sub-5060824', 'sub-5061803', 'sub-5066630', 'sub-5069529', 'sub-5074388', 'sub-5075090', 'sub-5078758', 'sub-5086093', 'sub-5088628', 'sub-5089212', 'sub-5094564', 'sub-5095553', 'sub-5107712', 'sub-5110387', 'sub-5111467', 'sub-5132313', 'sub-5133406', 'sub-5141407', 'sub-5146645', 'sub-5147292', 'sub-5147629', 'sub-5148531', 'sub-5152725', 'sub-5161493', 'sub-5161843', 'sub-5167064', 'sub-5169910', 'sub-5172259', 'sub-5174481', 'sub-5175426', 'sub-5190479', 'sub-5193459', 'sub-5195605', 'sub-5199897', 'sub-5204776', 'sub-5205920', 'sub-5207059', 'sub-5213480', 'sub-5214255', 'sub-5221314', 'sub-5225292', 'sub-5227219', 'sub-5238156', 'sub-5243208', 'sub-5244819', 'sub-5246168', 'sub-5248236', 'sub-5254707', 'sub-5255929', 'sub-5259572', 'sub-5260720', 'sub-5266774', 'sub-5268854', 'sub-5275590', 'sub-5277584', 'sub-5280127', 'sub-5284155', 'sub-5284757', 'sub-5285522', 'sub-5287379', 'sub-5288313', 'sub-5299738', 'sub-5300692', 'sub-5301387', 'sub-5301665', 'sub-5306331', 'sub-5307670', 'sub-5308272', 'sub-5309487', 'sub-5311128', 'sub-5312123', 'sub-5314453', 'sub-5314918', 'sub-5318996', 'sub-5321046', 'sub-5333219', 'sub-5337521', 'sub-5344927', 'sub-5356042', 'sub-5356225', 'sub-5357896', 'sub-5362869', 'sub-5362995', 'sub-5366453', 'sub-5367616', 'sub-5379362', 'sub-5381416', 'sub-5383726', 'sub-5385606', 'sub-5388640', 'sub-5388785', 'sub-5396295', 'sub-5407120', 'sub-5412743', 'sub-5413676', 'sub-5413852', 'sub-5414556', 'sub-5417377', 'sub-5419634', 'sub-5421372', 'sub-5423325', 'sub-5430117', 'sub-5430583', 'sub-5434420', 'sub-5436036', 'sub-5439511', 'sub-5439771', 'sub-5441585', 'sub-5443036', 'sub-5445233', 'sub-5447226', 'sub-5448135', 'sub-5451817', 'sub-5452360', 'sub-5456685', 'sub-5457995', 'sub-5461625', 'sub-5463855', 'sub-5466768', 'sub-5468006', 'sub-5469592', 'sub-5478211', 'sub-5483651', 'sub-5493301', 'sub-5500454', 'sub-5501206', 'sub-5501833', 'sub-5501960', 'sub-5505540', 'sub-5516440', 'sub-5519294', 'sub-5520211', 'sub-5526336', 'sub-5526572', 'sub-5532779', 'sub-5534037', 'sub-5534706', 'sub-5540456', 'sub-5549606', 'sub-5549981', 'sub-5551886', 'sub-5554939', 'sub-5555854', 'sub-5559865', 'sub-5567576', 'sub-5581661', 'sub-5585270', 'sub-5585805', 'sub-5588309', 'sub-5589600', 'sub-5591274', 'sub-5591418', 'sub-5592442', 'sub-5593078', 'sub-5596098', 'sub-5600529', 'sub-5601056', 'sub-5602380', 'sub-5605786', 'sub-5607168', 'sub-5609121', 'sub-5609431', 'sub-5611880', 'sub-5616277', 'sub-5619852', 'sub-5619899', 'sub-5621554', 'sub-5622421', 'sub-5624930', 'sub-5628318', 'sub-5631014', 'sub-5632024', 'sub-5642873', 'sub-5643081', 'sub-5644841', 'sub-5651992', 'sub-5654231', 'sub-5655606', 'sub-5656714', 'sub-5659756', 'sub-5662814', 'sub-5663992', 'sub-5676083', 'sub-5681563', 'sub-5682228', 'sub-5683004', 'sub-5686573', 'sub-5689399', 'sub-5691057', 'sub-5691618', 'sub-5695745', 'sub-5697322', 'sub-5702677', 'sub-5704903', 'sub-5704954', 'sub-5704989', 'sub-5710248', 'sub-5716665', 'sub-5718094', 'sub-5720673', 'sub-5723214', 'sub-5724408', 'sub-5730723', 'sub-5731172', 'sub-5731499', 'sub-5731679', 'sub-5741129', 'sub-5743157', 'sub-5745668', 'sub-5745702', 'sub-5756490', 'sub-5758679', 'sub-5759776', 'sub-5761201', 'sub-5761477', 'sub-5766354', 'sub-5766493', 'sub-5768522', 'sub-5769520', 'sub-5776312', 'sub-5778549', 'sub-5778557', 'sub-5782042', 'sub-5784625', 'sub-5787574', 'sub-5791907', 'sub-5793126', 'sub-5795406', 'sub-5798123', 'sub-5798883', 'sub-5799978', 'sub-5801259', 'sub-5803417', 'sub-5813144', 'sub-5817243', 'sub-5817924', 'sub-5833462', 'sub-5842889', 'sub-5848562', 'sub-5848719', 'sub-5849806', 'sub-5849978', 'sub-5857616', 'sub-5860615', 'sub-5861987', 'sub-5862579', 'sub-5864200', 'sub-5870034', 'sub-5872554', 'sub-5872987', 'sub-5877977', 'sub-5880748', 'sub-5881447', 'sub-5884233', 'sub-5884823', 'sub-5899486', 'sub-5904762', 'sub-5905199', 'sub-5910097', 'sub-5910394', 'sub-5910449', 'sub-5910611', 'sub-5911416', 'sub-5911570', 'sub-5916917', 'sub-5917010', 'sub-5920987', 'sub-5929220', 'sub-5936323', 'sub-5938579', 'sub-5944068', 'sub-5947763', 'sub-5948352', 'sub-5950797', 'sub-5952857', 'sub-5955885', 'sub-5960826', 'sub-5961157', 'sub-5962980', 'sub-5963255', 'sub-5964442', 'sub-5969185', 'sub-5970552', 'sub-5982312', 'sub-5982464', 'sub-5984905', 'sub-5992219', 'sub-5993620', 'sub-5999065', 'sub-6001450', 'sub-6004079', 'sub-6005827', 'sub-6008715', 'sub-6015125', 'sub-6018526', 'sub-6019541', 'sub-6019813', 'sub-6022316', 'sub-6023214']
    localpatients_list = os.listdir("/home/ajdelalo/projects/DLShapeAnalysis/UKB_Dataset")
    localpatients_list = sorted([i for i in localpatients_list if i.startswith("sub-")])

    # index_list = [1,4,8]
    index_list = [1,2,3,4]
    for i in index_list:
        index_patient = np.squeeze([j for j in range(len(patients_list)) if patients_list[j] == localpatients_list[i]])
        print(index_patient)
        if params["model_type"] == "separate":
            reconstruction, segmentation = ImplicitNetSeparateSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=index_patient, res_factors=(1,1,0.5))
        elif params["model_type"] == "shared":
            reconstruction, segmentation = ImplicitNetSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=index_patient, res_factors=(1,1,0.5))
        elif params["model_type"] == "mounted":
            reconstruction, segmentation = ImplicitNetMountedSegLatent.calculate_rec_seg(model, im_idx=i, index_patient=index_patient, res_factors=(1,1,0.5))
        else:
            raise ValueError("Unknown model type.")
        if not os.path.exists("results3"):
            os.mkdir("results3")
        nifti_seg = nib.Nifti1Image(segmentation, np.eye(4))
        nib.save(nifti_seg, f"./results3/segmentation{localpatients_list[i]}.nii.gz")
        nifti_image = nib.Nifti1Image(reconstruction, np.eye(4))
        nib.save(nifti_image, f"./results3/reconstruction{localpatients_list[i]}.nii.gz")

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
        weights_path, config_path = args.weights, args.config
        main_test(weights_path, config_path)
    else:
        raise ValueError("Unknown pipeline selected.")
