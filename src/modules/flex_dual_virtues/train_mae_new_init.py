import os
from src.utils.utils import setup_wandb_and_config, set_seed, init_rnd_seeds, is_rank0
from src.utils.training_utils import to_device
from src.data.datasets.mm_base import build_mm_datasets
from src.data.datasets.mae_dataset import MAEDataset, ExactAlignmentMAEDataset
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
from src.utils.masking_utils import get_mask_ratio
import numpy as np

class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs):
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every_n_epochs == 0:
            control.should_save = True
        else:
            if is_rank0():
                logger.info(f"Skipping save at epoch {state.epoch}, not a multiple of {self.save_every_n_epochs}")
            control.should_save = False
        return control

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


class MaskRatioCallBack(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        mask_ratio_update_strategy(
            masking_dict=mask_ratio_dict,
            epoch=state.epoch,
            total_epochs=args.num_train_epochs
        )
        if is_rank0():
            logger.info(f"New mask ratio: {mask_ratio_dict}")
            wandb.log({'train/lower_mask_ratio': mask_ratio_dict['mask_ratio'][0],
                        'train/higher_mask_ratio': mask_ratio_dict['mask_ratio'][1]},
                        step=state.global_step)

def build_mae_dataset(conf, datasets, split=None, inpainting_strategy='identity', inpainting_dict=None,
                      aligned=False, **kwargs):
    
    normalization = conf.image_info.normalization
    kwargs['normalization'] = normalization
    
    if not aligned:
        return MAEDataset(datasets=datasets, 
                        image_section_size=conf.image_info.image_section_size,
                        patch_size=conf.image_info.patch_size,
                        mask_ratio=mask_ratio_dict,
                        channel_fraction=conf.image_info.channel_fraction,
                        mask_strategy=conf.image_info.mask_strategy,
                        num_channels_to_keep=conf.image_info.num_channels_to_keep,
                        num_channels_to_full_mask=conf.image_info.num_channels_to_full_mask,
                        inpainting_strategy=inpainting_strategy,
                        split=split,
                        inpainting_dict=inpainting_dict,
                        use_fixed_drop_range=conf.image_info.use_fixed_drop_range,
                        **kwargs
                        )
    else:
        return ExactAlignmentMAEDataset(datasets=datasets, 
                        image_section_size=conf.image_info.image_section_size,
                        patch_size=conf.image_info.patch_size,
                        mask_ratio=mask_ratio_dict,
                        mask_strategy=conf.image_info.mask_strategy,
                        channel_fraction=conf.image_info.channel_fraction,
                        num_channels_to_keep=conf.image_info.num_channels_to_keep,
                        num_channels_to_full_mask=conf.image_info.num_channels_to_full_mask,
                        inpainting_strategy=inpainting_strategy,
                        split=split,
                        inpainting_dict=inpainting_dict,
                        use_fixed_drop_range=conf.image_info.use_fixed_drop_range,
                        **kwargs
                        )

class CustomTrainer(Trainer):
    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
        outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        recon_multiplex, recon_he = outputs
        if self.conf.training.loss_to_use == 'DualMAELossWithEdge':
            loss_total, metrics = mae_loss_fn(recon_multiplex, recon_he, multiplex, he, multiplex_mask=multiplex_mask) #, he_mask)
        else:
            loss_total, metrics = mae_loss_fn(recon_multiplex, recon_he, multiplex, he)
        if self.conf.training.loss_to_use == 'DualMAELossWithEdge' and is_rank0() and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            loss_tracker = {}
            if use_ssim:
                loss_tracker['custom/loss_ssim'] = metrics['loss_ssim']
            if use_sobel:
                loss_tracker['custom/loss_sobel'] = metrics['loss_sobel']
            if use_fft:
                loss_tracker['custom/loss_fft'] = metrics['loss_fft']
            loss_tracker['custom/loss_mse_masked'] = metrics['loss_mse_masked']
            loss_tracker['custom/loss_mse_all'] = metrics['loss_mse_all']
            logger.debug(
                f"Epoch {self.state.epoch:.4f} - Step {self.state.global_step} - Loss: {loss_total:.4f} - "
                f"Loss MSE: {metrics['loss_mse_all']:.4f} Loss MSE_Mask {metrics['loss_mse_masked']:.4f} - Loss SSIM: {metrics['loss_ssim']:.4f} - "
                f"Loss Sobel: {metrics['loss_sobel']:.4f} - Loss FFT: {metrics['loss_fft']:.4f}"
            )
            wandb.log(loss_tracker)



        return (loss_total, outputs) if return_outputs else loss_total
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        if DISABLE_EVAL:
            return {}
        else:
            eval_dataloader = {'test': self.get_eval_dataloader('test')}
            if self.conf.inpainting.track:
                inpaint_test_dataloader = self.get_eval_dataloader('test_inpaint')
                eval_dataloader['test_inpaint'] = inpaint_test_dataloader

            loss_metrics = evaluate_fn(self.conf, self.model, eval_dataloader, self.state.epoch)
            
            return loss_metrics


def custom_collate_fn(batch):
    return list(zip(*batch))

def train_mae(conf):
    if hasattr(conf, 'datasets_kwargs'):
        data_kwargs = conf.datasets_kwargs
    else:
        data_kwargs = {}
    
    mm_datasets = build_mm_datasets(conf, **data_kwargs)

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
    mae_model.compile_rope()

    if is_rank0():
        logger.info(f'Model has {sum(p.numel() for p in mae_model.parameters() if p.requires_grad)} trainable parameters')
    mae_model.cuda()

    training_args = TrainingArguments(
        output_dir=f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints',
        num_train_epochs=conf.training.epochs,
        per_device_train_batch_size=conf.training.batch_size,
        per_device_eval_batch_size=conf.training.eval_boost * conf.training.batch_size if not DISABLE_EVAL else conf.training.batch_size,
        eval_strategy="epoch" if not DISABLE_EVAL else "no",
        eval_steps=conf.training.eval_freq, # we do epoch wise eval
        save_strategy="epoch",  # Save the model at the end of each epoch
        # save_steps=200,  # Save the model every 100 steps
        save_total_limit=5,  # Only keep the best 5 models
        logging_dir=f'{conf.experiment.dir}/{conf.experiment.name}/logs',
        logging_strategy="steps",
        logging_steps=100,
        fp16=conf.training.fp16,
        bf16=conf.training.bf16,
        dataloader_num_workers=conf.training.num_workers,
        report_to="wandb",
        run_name=conf.experiment.name,
        gradient_accumulation_steps=conf.training.grad_accumulation,
        learning_rate=conf.training.lr,
        lr_scheduler_type=conf.training.lr_scheduler_type,
        weight_decay=conf.training.weight_decay,
        ddp_find_unused_parameters=True,
        warmup_ratio=(conf.training.warmup_epochs / conf.training.epochs) if conf.training.warmup_epochs > 0 else 0.0,
        # auto_find_batch_size=True,
    )

    if is_rank0():
        logger.info(f"Training with warmup ratio {training_args.warmup_ratio} and gradient accumulation {training_args.gradient_accumulation_steps}")

    trainer = CustomTrainer(
        conf=conf, 
        model=mae_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        compute_metrics=compute_metrics, 
        data_collator=custom_collate_fn
    )

    if conf.inpainting.track:
        trainer.add_callback(InpaintingUpdateCallback())
        if is_rank0():
            logger.info('Inpainting update strategy callback added')
    

    trainer.add_callback(MaskRatioCallBack())

    trainer.add_callback(SaveEveryNEpochsCallback(save_every_n_epochs=25))

    if is_rank0():
        logger.info('Mask ratio callback added')

    if is_rank0():
        logger.info('Starting training')
    
    trainer.train(
        resume_from_checkpoint=conf.resume_from_checkpoint
    )

    if is_rank0():
        logger.info('Finished training')


def evaluate_mse(conf, model, test_dataloader, epoch):
    model.eval()
    all_metrics = []
    test_dataloader = test_dataloader['test']
    for inputs in test_dataloader:

        multiplex, he, multiplex_mask, he_mask, multiplex_channels, _, _ = inputs

        multiplex = to_device(multiplex, 'cuda')
        he = to_device(he, 'cuda')
        multiplex_mask = to_device(multiplex_mask, 'cuda')
        he_mask = to_device(he_mask, 'cuda')
        multiplex_channels = to_device(multiplex_channels, 'cuda')

        # new virtues
        with torch.no_grad():
            # not necessary since this is done by HF Trainer
            # with torch.amp.autocast('cuda', enabled=conf.training.fp16):
            outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        recon_multiplex, recon_he = outputs
        loss_total, mae_metrics = eval_mae_loss_fn(recon_multiplex, recon_he, multiplex, he, multiplex_mask=multiplex_mask)
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
            # with torch.amp.autocast('cuda', enabled=conf.training.fp16):
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

    # with Sobel 
    elif conf.training.loss_to_use == 'DualMAELossWithEdge':
        from modules.losses import DualMAELossWithEdge
        if hasattr(conf.training, 'use_ssim'):
            use_ssim = conf.training.use_ssim
        else:
            use_ssim = False

        if hasattr(conf.training, 'use_sobel'):
            use_sobel = conf.training.use_sobel
        else:
            use_sobel = False

        if hasattr(conf.training, 'use_fft'):
            use_fft = conf.training.use_fft
        else:
            use_fft = False

        if hasattr(conf.training, 'use_masked_mse'):
            use_masked_mse = conf.training.use_masked_mse
        else:
            use_masked_mse = True

        if is_rank0():
            logger.debug(f'Using loss with edge: use_ssim={use_ssim}, use_sobel={use_sobel}, use_fft={use_fft}, use_masked_mse={use_masked_mse}')


        mae_loss_fn = DualMAELossWithEdge(alpha_imc=1.0, alpha_he=1.0,
                                            use_ssim=use_ssim, use_sobel=use_sobel, use_fft=use_fft)
        
        if not DISABLE_EVAL:
            if conf.training.evaluate_dataset_wise:
                eval_mae_loss_fn = DualDatasetWiseMAELoss(alpha_imc=1.0, alpha_he=1.0)
                evaluate_fn = evaluate_dataset_wise
            else:
                eval_mae_loss_fn = DualMAELossWithEdge(alpha_imc=1.0, alpha_he=1.0)
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
    if hasattr(conf, 'masking'):
        if hasattr(conf.masking, 'epoch_threshold'):
            ec = conf.masking.epoch_threshold
        else:
            ec = None
        if 'alpha' in conf.masking.update_strategy:
            initial_alpha = conf.masking.initial_alpha
            final_alpha = conf.masking.final_alpha
        else:
            initial_alpha = None
            final_alpha = None
        mask_ratio_update_strategy, mask_ratio_dict = get_mask_ratio(conf.masking.update_strategy, list(conf.masking.initial_ratio), list(conf.masking.final_ratio),
                                                                    conf.training.epochs, epoch_threshold=ec, initial_alpha=initial_alpha, final_alpha=final_alpha)                                       


    train_mae(conf)