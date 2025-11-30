import os
from src.utils.utils import setup_wandb_and_config, set_seed, init_rnd_seeds, is_rank0
from src.utils.training_utils import to_device
from src.dataset.datasets.mm_base import build_mm_datasets
from src.dataset.datasets.mae_dataset import MAEDataset, build_mae_dataset
from src.modules.flex_dual_virtues.flex_dual_virtues_new_init import build_flex_dual_virtues
from src.utils.marker_utils import load_marker_embeddings
from torch.utils.data import random_split
from src.modules.losses import MAELoss, DualMAELoss, DualHuberLoss, DualDatasetWiseMAELoss
import torch
from einops import rearrange
import wandb
from loguru import logger
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments, TrainerCallback
from collections import defaultdict
from src.utils.inpainting_utils import get_inpainting_update_strategy
import numpy as np

import torch
import torch.nn as nn

from calflops.utils import generate_transformer_input
from calflops.utils import flops_to_string
from calflops.utils import macs_to_string
from calflops.utils import params_to_string

from calflops.calculate_pipline import CalFlopsPipline

def my_calculate_flops(model,
                    input_shape=None,
                    transformer_tokenizer=None,
                    args=[],
                    kwargs={},
                    forward_mode="forward",
                    include_backPropagation=False,
                    compute_bp_factor=2.0,
                    print_results=True,
                    print_detailed=True,
                    output_as_string=True,
                    output_precision=2,
                    output_unit=None,
                    ignore_modules=None):
    
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Args:
        model ([torch.nn.Module]): The model of input must be a PyTorch model.
        input_shape (tuple, optional): Input shape to the model. If args and kwargs is empty, the model takes a tensor with this shape as the only positional argument. Default to [].
        transformers_tokenizer (None, optional): Transforemrs Toekenizer must be special if model type is transformers and args、kwargs is empty. Default to None
        args (list, optional): list of positional arguments to the model, such as bert input args is [input_ids, token_type_ids, attention_mask]. Default to []
        kwargs (dict, optional): dictionary of keyword arguments to the model, such as bert input kwargs is {'input_ids': ..., 'token_type_ids':..., 'attention_mask':...}. Default to {}
        forward_mode (str, optional): To determine the mode of model inference, Default to 'forward'. And use 'generate' if model inference uses model.generate().
        include_backPropagation (bool, optional): Decides whether the final return FLOPs computation includes the computation for backpropagation.
        compute_bp_factor (float, optional): The model backpropagation is a multiple of the forward propagation computation. Default to 2.
        print_results (bool, optional): Whether to print the model profile. Defaults to True.
        print_detailed (bool, optional): Whether to print the detailed model profile. Defaults to True.
        output_as_string (bool, optional): Whether to print the output as string. Defaults to True.
        output_precision (int, optional) : Output holds the number of decimal places if output_as_string is True. Default to 2.
        output_unit (str, optional): The unit used to output the result value, such as T, G, M, and K. Default is None, that is the unit of the output decide on value.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Example:
    .. code-block:: python
    from calflops import calculate_flops

    # Deep Learning Model, such as alexnet.
    from torchvision import models

    model = models.alexnet()
    batch_size = 1
    flops, macs, params = calculate_flops(model=model, 
                                          input_shape=(batch_size, 3, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Alexnet FLOPs:1.4297 GFLOPS   MACs:714.188 MMACs   Params:61.1008 M 

    # Transformers Model, such as bert.
    from transformers import AutoModel
    from transformers import AutoTokenizer
    batch_size = 1
    max_seq_length = 128
    model_name = "hfl/chinese-roberta-wwm-ext/"
    model_save = "../pretrain_models/" + model_name
    model = AutoModel.from_pretrained(model_save)
    tokenizer = AutoTokenizer.from_pretrained(model_save)
    flops, macs, params = calculate_flops(model=model, 
                                          input_shape=(batch_size, max_seq_length),
                                          transformer_tokenizer=tokenizer)
    print("Bert(hfl/chinese-roberta-wwm-ext) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Bert(hfl/chinese-roberta-wwm-ext) FLOPs:22.36 GFLOPS   MACs:11.17 GMACs   Params:102.27 M 

    # Large Languase Model, such as llama2-7b.
    from transformers import LlamaTokenizer
    from transformers import LlamaForCausalLM
    batch_size = 1
    max_seq_length = 128
    model_name = "llama2_hf_7B"
    model_save = "../model/" + model_name
    model = LlamaForCausalLM.from_pretrained(model_save)
    tokenizer = LlamaTokenizer.from_pretrained(model_save)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=(batch_size, max_seq_length),
                                          transformer_tokenizer=tokenizer)
    print("Llama2(7B) FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    #Llama2(7B) FLOPs:1.7 TFLOPS   MACs:850.00 GMACs   Params:6.74 B 

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """

    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    # assert transformers_tokenizer and auto_generate_transformers_input and "transformers" in str(type(model)), "The model must be a transformers model if args of auto_generate_transformers_input is True and transformers_tokenizer is not None"
    model.eval()

    is_Transformer = True if "transformers" in str(type(model)) else False
    
    calculate_flops_pipline = CalFlopsPipline(model=model, 
                                                  include_backPropagation=include_backPropagation,
                                                  compute_bp_factor=compute_bp_factor)
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)
    
    device = next(model.parameters()).device
    model = model.to(device)
    
    if input_shape is not None:
        assert len(args) == 0 and len(kwargs) == 0, "args and kwargs must be empty value if input_shape is not None, then will be generate random input by inpust_shape"
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        
        if transformer_tokenizer is None:  # model is not transformers model
            assert is_Transformer is False, "the model is must not transformer model if input_shape is not None and transformer_tokenizer is None"
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape, ),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape, ))
            args = [input]
        else:
            assert len(input_shape) == 2, "the format of input_shape must be (batch_size, seq_len) if model is transformers model and auto_generate_transformers_input if True"
            kwargs = generate_transformer_input(input_shape=input_shape,
                                                model_tokenizer=transformer_tokenizer,
                                                device=device)
    else:
        assert transformer_tokenizer or (len(args) > 0 or len(kwargs) > 0),  "input_shape or args or kwargs one of there parameters must specified if auto_generate_input is False"
        if transformer_tokenizer:
            kwargs = generate_transformer_input(input_shape=None,
                                                model_tokenizer=transformer_tokenizer,
                                                device=device)
    
    if kwargs:
        for key, value in kwargs.items():
            # kwargs[key] = value.to(device)
            kwargs[key] = to_device(value, device)

        if forward_mode == 'forward':
            with torch.amp.autocast('cuda', enabled=True):
                _ = model(*args, **kwargs)
        if forward_mode == 'generate':
            _ = model.generate(*args, **kwargs)
    else:
        for index in range(len(args)):
            args[index] = args[index].to(device)
            
        if forward_mode == 'forward':
            _ = model(*args)
        if forward_mode == 'generate':
            _ = model.generate(*args)

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()
    
    if print_results:
        return_print = calculate_flops_pipline.print_model_pipline(units=output_unit,
                                                    precision=output_precision,
                                                    print_detailed=print_detailed)

    calculate_flops_pipline.end_flops_calculate()
   
    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor) 
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return flops_to_string(flops, units=output_unit, precision=output_precision),\
               macs_to_string(macs, units=output_unit, precision=output_precision),  \
               params_to_string(params, units=output_unit, precision=output_precision)
    
    return flops, macs, params


class InpaintingUpdateCallback(TrainerCallback):
    """
    A callback to update inpainting strategy
    """
    def on_epoch_begin(self, args, state, control, **kwargs):
        inpainting_update_strategy(
            inpainting_dict=inpainting_dict,
            epoch=state.epoch,
            total_epochs=args.num_train_epochs
        )

def compute_loss(model, inputs, return_outputs=False, **kwargs):
        multiplex, he, multiplex_mask, he_mask, multiplex_channels, _, _ = inputs

        multiplex = to_device(multiplex, 'cuda')
        he = to_device(he, 'cuda')
        multiplex_mask = to_device(multiplex_mask, 'cuda')
        he_mask = to_device(he_mask, 'cuda')
        multiplex_channels = to_device(multiplex_channels, 'cuda')

        # #  old virtues
        # outputs = model.forward(multiplex, multiplex_channels, mask=multiplex_mask)
        # target_img = torch.concat(multiplex, dim=0)
        # target_img = rearrange(target_img, 'c (H p) (W q) -> c H W (p q)', p=self.conf.image_info.patch_size, q=self.conf.image_info.patch_size)
        # multiplex_mask = torch.concat(multiplex_mask, dim=0)
        # mae_loss_fn = MAELoss(predict_all=True, alpha_fft=0.0)
        # loss_total, mae_metrics = mae_loss_fn(outputs, target_img, multiplex_mask)

        # new virtues 
        # outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        # recon_multiplex, recon_he = outputs
        # loss_total, metrics = mae_loss_fn(recon_multiplex, recon_he, multiplex, he)

        # return (loss_total, outputs) if return_outputs else loss_total
        my_calculate_flops(model, kwargs={
                'multiplex': multiplex,
                'he': he,
                'channel_ids': multiplex_channels,
                'multiplex_mask': multiplex_mask,
                'he_mask': he_mask
            }
        )

def custom_collate_fn(batch):
    return list(zip(*batch))

def train_mae(conf):
    mm_datasets = build_mm_datasets(conf)

    if is_rank0():
        logger.info(f'Loaded {len(mm_datasets)} datasets:')
        for ds in mm_datasets:
            logger.info(f'{ds.name} with {len(ds)} samples.')


    
    train_dataset = build_mae_dataset(conf, mm_datasets, split='train',
                                        inpainting_strategy=conf.inpainting.mask_strategy,
                                        inpainting_dict=inpainting_dict,
                                        aligned=conf.training.train_on_aligned)
    if not DISABLE_EVAL:
        test_dataset = build_mae_dataset(conf, mm_datasets, split='test',
                                            inpainting_strategy='identity',
                                            inpainting_dict=None)
        trainer_eval_dataset = {'test': test_dataset}
        if conf.inpainting.track:
            inpaint_test_dataset = build_mae_dataset(conf, mm_datasets, split='test',
                                                        inpainting_strategy=conf.inpainting.mask_strategy,
                                                        inpainting_dict=inpainting_dict,
                                                        aligned=True)
            trainer_eval_dataset['test_inpaint'] = inpaint_test_dataset
        
    else:
        test_dataset = []
        trainer_eval_dataset = None
    if is_rank0():
        logger.info(f'Train dataset has {len(train_dataset)} samples, test dataset has {len(test_dataset)} samples')

    esm_embeddings = load_marker_embeddings(conf.marker_embedding_dir)
    if is_rank0():
        logger.info(f'Loaded ESM embeddings of shape {esm_embeddings.shape}')

    # mae_model = build_virtues_mae(conf, esm_embeddings)

    kwargs = {'unimodal_training': UNIMODAL_TRAINING}
    if hasattr(conf, 'kwargs') and conf.kwargs is not None:
        kwargs = conf.kwargs
        if is_rank0():
            logger.info(f'Using kwargs: {kwargs}')

    mae_model = build_flex_dual_virtues(conf, esm_embeddings, **kwargs)

    if is_rank0():
        logger.info(f'Model has {sum(p.numel() for p in mae_model.parameters() if p.requires_grad)} trainable parameters')
    mae_model.cuda()

    
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.training.batch_size,
        shuffle=True,
        num_workers=conf.training.num_workers,
        collate_fn=custom_collate_fn,
    )

    batch = next(iter(train_dl))
    compute_loss(mae_model, batch)



    if is_rank0():
        logger.info('Finished flop calculation')


def evaluate_mse(conf, model, test_dataloader, epoch):
    model.eval()
    all_metrics = []
    for inputs in test_dataloader:

        multiplex, he, multiplex_mask, he_mask, multiplex_channels, _, _ = inputs

        multiplex = to_device(multiplex, 'cuda')
        he = to_device(he, 'cuda')
        multiplex_mask = to_device(multiplex_mask, 'cuda')
        he_mask = to_device(he_mask, 'cuda')
        multiplex_channels = to_device(multiplex_channels, 'cuda')

        # new virtues
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        recon_multiplex, recon_he = outputs
        loss_total, mae_metrics = eval_mae_loss_fn(recon_multiplex, recon_he, multiplex, he)
        all_metrics.append(mae_metrics)

    sum_metrics = dict()
    num_metrics = dict()
    for m in all_metrics:
        for k, v in m.items():
            if k not in sum_metrics:
                sum_metrics[k] = v
                num_metrics[k] = 1
            else:
                sum_metrics[k] += v
                num_metrics[k] += 1
    avg_metrics = {f'test_{k}': sum_metrics[k] / num_metrics[k] for k in sum_metrics.keys()} 
    keys = sorted(avg_metrics.keys())
    avg_metrics = sync_metrics(avg_metrics, ['test_loss_he', 'test_loss_multiplex', 'test_loss_total'])
    avg_metrics["epoch"] = epoch
    if is_rank0():
        logger.info(avg_metrics)
        wandb.log(avg_metrics)
    return avg_metrics

@logger.catch
def evaluate_dataset_wise(conf, model, test_dataloader_dict, epoch):
    model.eval()
    all_multiplex_metrics = defaultdict(float)
    all_he_metrics = defaultdict(float)
    all_multiplex_dataset_counter = defaultdict(int)
    all_he_dataset_counter = defaultdict(int)
    all_inpaint_metrics = defaultdict(float)
    all_inpaint_counter = defaultdict(int)

    test_dataloader = test_dataloader_dict['test']
    if is_rank0():
        logger.info(f'Evaluating on {len(test_dataloader)} batches')
    for inputs in test_dataloader:

        multiplex, he, multiplex_mask, he_mask, multiplex_channels, tid, inpaint_mask = inputs

        multiplex = to_device(multiplex, 'cuda')
        he = to_device(he, 'cuda')
        multiplex_mask = to_device(multiplex_mask, 'cuda')
        he_mask = to_device(he_mask, 'cuda')
        multiplex_channels = to_device(multiplex_channels, 'cuda')

        # new virtues
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        recon_multiplex, recon_he = outputs
        he_metrics, multiplex_metrics, _, he_dataset_counter, multiplex_dataset_counter, _ = eval_mae_loss_fn(recon_multiplex, recon_he, multiplex, he, tid, inpaint_mask, track_inpainting=False)
        for k, v in he_metrics.items():
            all_he_metrics[k] += v
        for k, v in multiplex_metrics.items():
            all_multiplex_metrics[k] += v
        for k, v in he_dataset_counter.items():
            all_he_dataset_counter[k] += v
        for k, v in multiplex_dataset_counter.items():
            all_multiplex_dataset_counter[k] += v
    

    if conf.inpainting.track:
        test_inpaint_dataloader = test_dataloader_dict['test_inpaint']
        if is_rank0():
            logger.info(f'Evaluating inpainting on {len(test_inpaint_dataloader)} batches')
        for inputs in test_inpaint_dataloader:

            multiplex, he, multiplex_mask, he_mask, multiplex_channels, tid, inpaint_mask = inputs
            assert np.all(inpaint_mask)

            multiplex = to_device(multiplex, 'cuda')
            he = to_device(he, 'cuda')
            multiplex_mask = to_device(multiplex_mask, 'cuda')
            he_mask = to_device(he_mask, 'cuda')
            multiplex_channels = to_device(multiplex_channels, 'cuda')

            # new virtues
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=conf.training.fp16):
                    outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
            recon_multiplex, recon_he = outputs
            _, _, inpainting_metrics, _, _, inpainting_counter  = eval_mae_loss_fn(recon_multiplex, recon_he, multiplex, he, tid, inpaint_mask, track_inpainting=True)

            for k, v in inpainting_metrics.items():
                all_inpaint_metrics[k] += v
            for k, v in inpainting_counter.items():
                all_inpaint_counter[k] += v


    avg_metrics = defaultdict(float)
    total_he = 0
    total_multiplex = 0
    total_inpaint = 0
    for k in all_he_metrics.keys():
        avg_metrics[f'test_he_{k}'] = all_he_metrics[k] / all_he_dataset_counter[k]
        avg_metrics[f'test_he_total'] += all_he_metrics[k]
        total_he += all_he_dataset_counter[k]
    for k in all_multiplex_metrics.keys():
        avg_metrics[f'test_multiplex_{k}'] = all_multiplex_metrics[k] / all_multiplex_dataset_counter[k]
        avg_metrics[f'test_multiplex_total'] += all_multiplex_metrics[k]
        total_multiplex += all_multiplex_dataset_counter[k]

    if conf.inpainting.track:
        for k in all_inpaint_metrics.keys():
            avg_metrics[f'test_inpaint_{k}'] = all_inpaint_metrics[k] / all_inpaint_counter[k]
            avg_metrics[f'test_inpaint_total'] += all_inpaint_metrics[k]
            total_inpaint += all_inpaint_counter[k]
        


    if total_he > 0:
        avg_metrics['test_he_total'] /= total_he
    if total_multiplex > 0:
        avg_metrics['test_multiplex_total'] /= total_multiplex
    if conf.inpainting.track and total_inpaint > 0:
        avg_metrics['test_inpaint_total'] /= total_inpaint
        

    keys = sorted(avg_metrics.keys())
    avg_metrics = sync_metrics(avg_metrics, keys)
    avg_metrics["epoch"] = epoch
    if is_rank0():
        logger.info(avg_metrics)
        wandb.log(avg_metrics)
    return avg_metrics


def sync_metrics(metrics, list_of_metrics):
    """
    Syncs metrics across all ranks by averaging them. Accounts for the possibility that a metric is not present on all ranks.
    metrics: dict with metrics
    list_of_metrics: list of metrics to sync; ensures that metrics not present on a single rank are properly handled
    """
    metrics_to_sync = torch.tensor([metrics.get(k, 0.0) for k in sorted(list_of_metrics)]).cuda()
    mask = []
    for k in sorted(list_of_metrics):
        if k in metrics:
            mask.append(1.0)
        else:
            mask.append(0.0)
    mask_to_sync = torch.tensor(mask).cuda()
    torch.distributed.all_reduce(metrics_to_sync, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(mask_to_sync, op=torch.distributed.ReduceOp.SUM)
    metrics = {k: metrics_to_sync[i].item() / max(mask_to_sync[i].item(),1) for i, k in enumerate(sorted(list_of_metrics))}
    return metrics

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {}

if __name__ == "__main__":
    conf = OmegaConf.load("configs/base_config.yaml")

    cli_conf = OmegaConf.from_cli()
    if hasattr(cli_conf, 'dataset_config') and cli_conf.dataset_config is not None:
        dataset_conf = OmegaConf.load(cli_conf.dataset_config)
        conf = OmegaConf.merge(conf, dataset_conf)
    elif hasattr(conf, 'dataset_config') and conf.dataset_config is not None:
        dataset_conf = OmegaConf.load(conf.dataset_config)
        conf = OmegaConf.merge(conf, dataset_conf)

    if hasattr(cli_conf, 'additional_config') and cli_conf.additional_config is not None:
        additional_conf = OmegaConf.load(cli_conf.additional_config)
        conf = OmegaConf.merge(conf, additional_conf)
    elif hasattr(conf, 'additional_config') and conf.additional_config is not None:
        additional_conf = OmegaConf.load(conf.additional_config)
        conf = OmegaConf.merge(conf, additional_conf)

    if hasattr(cli_conf, 'model_config') and cli_conf.model_config is not None:
        model_conf = OmegaConf.load(cli_conf.model_config)
        conf = OmegaConf.merge(conf, model_conf)
    elif hasattr(conf, 'model_config') and conf.model_config is not None:
        model_conf = OmegaConf.load(conf.model_config)
        conf = OmegaConf.merge(conf, model_conf)
    else:
        model_conf = OmegaConf.load('configs/model_configs/base.yaml')
        conf = OmegaConf.merge(conf, model_conf)

    conf = OmegaConf.merge(conf, cli_conf)


    if hasattr(conf, 'rcp') and conf.rcp:
        # The run is on RCP cluster
        conf.marker_embedding_dir = '/mnt/aimm/scratch/datasets/marker_embeddings/esm2_t30_150M_UR50D'
        conf.experiment.dir = '/mnt/aimm/scratch/shr/mmvirtues/outputs'


    if is_rank0():
        logger.info(OmegaConf.to_yaml(conf))

    os.makedirs(conf.experiment.dir, exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/wandb', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/logs', exist_ok=True)

    if is_rank0():
        logger.add(f'{conf.experiment.dir}/{conf.experiment.name}/logs/train.log')
        logger.info(f"Starting experiment {conf.experiment.name}")

    conf = setup_wandb_and_config(conf, tags=['new_init'])

    set_seed(conf.training.seed)
    if hasattr(conf.image_info, 'unimodal') and conf.image_info.unimodal:
        UNIMODAL_TRAINING = True
        if is_rank0():
            logger.info('Unimodal training')
    else:
        UNIMODAL_TRAINING = False

    DISABLE_EVAL = conf.training.disable_eval
    if DISABLE_EVAL:
        if is_rank0():
            logger.info('Evaluation disabled')


    if conf.training.loss_to_use == 'DualMAELoss':
        mae_loss_fn = DualMAELoss(alpha_imc=1.0, alpha_he=1.0)
        if not DISABLE_EVAL:
            if conf.training.evaluate_dataset_wise:
                eval_mae_loss_fn = DualDatasetWiseMAELoss(alpha_imc=1.0, alpha_he=1.0)
                evaluate_fn = evaluate_dataset_wise
            else:
                eval_mae_loss_fn = DualMAELoss(alpha_imc=1.0, alpha_he=1.0)
                evaluate_fn = evaluate_mse
    elif conf.training.loss_to_use == 'DualHuberLoss':
        mae_loss_fn = DualHuberLoss(alpha_imc=1.0, alpha_he=1.0)
        if not DISABLE_EVAL:
            eval_mae_loss_fn = DualHuberLoss(alpha_imc=1.0, alpha_he=1.0)
            evaluate_fn = evaluate_mse
    else:
        raise NotImplementedError(f"Loss {conf.training.loss_to_use} not implemented")
    

    max_num_modalities = 0
    for ds in conf.datasets:
        max_num_modalities = max(len(conf.datasets[ds]['modalities']), max_num_modalities)
    
    if max_num_modalities == 1:
        conf.inpainting.track = False

    if conf.training.train_on_aligned:
        if is_rank0():
            logger.info('Training on aligned data')


    # We want inpainting_update_strategy to be global so that it can be used in the evaluation function
    inpainting_update_strategy, inpainting_dict = get_inpainting_update_strategy(conf.inpainting.update_strategy)

    train_mae(conf)