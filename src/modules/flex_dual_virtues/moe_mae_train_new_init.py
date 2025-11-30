import os
from src.utils.utils import setup_wandb_and_config, set_seed, init_rnd_seeds, is_rank0, load_checkpoint_safetensors
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
import pickle
from src.modules.flex_dual_virtues.layers.transformers_flashattention import TransformerEncoderBlock
from src.modules.flex_dual_virtues.layers.basics import build_feedforward
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """
    Simplest MoE implementation with a linear router and softmax over experts.

    Note that in this implementation, we simply loop over the experts and
    aggregate the results. This is not the most efficient way to do it, but
    it also avoids the large memory overhead _and_ has no token dropping
    (because we do not need the capacity factor).
    """

    def __init__(self, d_model, dim_feedforward, activation, dropout, moe_num_experts=4, moe_num_experts_per_tok=2, moe_softmax_order="softmax_topk"):
        super().__init__()
        assert moe_num_experts > 0
        self.experts = nn.ModuleList(
            [build_feedforward(d_model, dim_feedforward, activation, dropout) for _ in range(moe_num_experts)]
        )
        self.router = nn.Linear(d_model, moe_num_experts, bias=False)
        self.top_k = moe_num_experts_per_tok
        self.softmax_order = moe_softmax_order

    def forward(self, inputs: torch.Tensor):
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        # [batch_size * sequence_length, num_experts]
        router_logits = self.router(inputs_squashed)

        # note that selected experts will be the same for all orders:
        # softmax doesnt change top-k, but the weights are different
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        results = torch.zeros_like(inputs_squashed)
        # naive looping over experts
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            output = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output

        # return results and router logits (for aux loss calculation later)
        # return results.view_as(inputs), {
        #     "router_logits": router_logits,
        #     "selected_experts": selected_experts,
        # }
        return results.view_as(inputs), router_logits

    # def _maintain_float32_expert_bias(self):
    #     if hasattr(self, 'expert_bias') and self.expert_bias is not None:
    #         if self.expert_bias.dtype != torch.float32:
    #             self.expert_bias.data = self.expert_bias.data.to(torch.float32)

def router_z_loss(router_logits: torch.Tensor) -> float:
    """Compute router z-loss.

     The router z-loss was introduced in Designing Effective Sparse Expert Models
     (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
     small in an effort to improve stability.

    Args:
      router_logits: <float>[batch_size * sequence_length, num_experts]
        router logits

    Returns:
      Scalar router z-loss.
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss, dtype=torch.float32) / (num_tokens)

def get_bool_mask_on_moe(pattern, HV_FLAG, MOE_FREQ):
    mask = [False] * len(pattern)
    if not HV_FLAG:
        if len(pattern) < MOE_FREQ:
            return mask
        for i in range(1, len(pattern)+1):
            if i % MOE_FREQ == 0:
                mask[i-1] = True
        return mask
    else:
        # consider now in blocks of 2:
        if len(pattern) < 2 * MOE_FREQ:
            return mask
        for i in range(1, len(pattern)+1, 2):
            if (i+1) % (2*MOE_FREQ) == 0:
                mask[i-1] = True
                mask[i] = True
            
        return mask



def convert_feedforward_of_transformer_to_moe(
        model,
        num_experts,
        num_experts_per_token,
        softmax_order,
        pattern,
        HV_FLAG=False,
):
    # loop over all instances of the model to find TransformerEncoderBlock
    if HV_FLAG:
        if is_rank0():
            logger.info('Converting to MoE using HV pattern')
    else:
        if is_rank0():
            logger.info('Converting to MoE using full pattern')

    transformer_count = 0
    bool_mask = get_bool_mask_on_moe(pattern, HV_FLAG, MOE_FREQ)
    for name, module in model.named_modules():
        if isinstance(module, TransformerEncoderBlock):
            # replace feedforward with MoE
            if bool_mask[transformer_count]:
                moe_layer = MoE(
                    d_model=module.d_model,
                    dim_feedforward=module.dim_feedforward,
                    activation=module.activation,
                    dropout=module.dropout,
                    moe_num_experts=num_experts,
                    moe_num_experts_per_tok=num_experts_per_token,
                    moe_softmax_order=softmax_order,
                ).to(device='cuda', dtype=torch.float32)
                old_feedforward_dict = module.feedforward.state_dict()
                module.feedforward = moe_layer
                # load into each expert the weights of the old feedforward
                for i in range(num_experts):
                    module.feedforward.experts[i].load_state_dict(old_feedforward_dict)
                # set the router weights to 0
                module.feedforward.router.weight.data.fill_(0)
                # set the router bias to 0
                if module.feedforward.router.bias is not None:
                    module.feedforward.router.bias.data.fill_(0)
            transformer_count += 1

def convert_all_modules_to_moe(model, num_experts, num_experts_per_token, softmax_order):
    sep_enc = conf.model.separate_encoders if hasattr(conf.model, 'separate_encoders') else False
    if sep_enc:
        if is_rank0():
            logger.info('Converting separate encoders to MoE')
        

        # multiplex encoder
        if is_rank0():
            logger.info('Converting multiplex encoder to MoE')
        hv_flag_multiplex_encoder = conf.model.separate_encoder_pattern_multiplex if hasattr(conf.model, 'separate_encoder_pattern_multiplex') else "hvhv"
        convert_feedforward_of_transformer_to_moe(model.encoder.multiplex_encoder, num_experts, num_experts_per_token, softmax_order,
                                                  HV_FLAG='hv' in hv_flag_multiplex_encoder, pattern=hv_flag_multiplex_encoder)
        # HE encoder
        hv_flag_he_encoder = conf.model.separate_encoder_pattern_he if hasattr(conf.model, 'separate_encoder_pattern_he') else "hvhv"
        if hasattr(model.encoder, 'he_encoder'):
            if is_rank0():
                logger.info('Converting HE encoder to MoE')
            convert_feedforward_of_transformer_to_moe(model.encoder.he_encoder, num_experts, num_experts_per_token, softmax_order,
                                                      HV_FLAG='hv' in hv_flag_he_encoder, pattern=hv_flag_he_encoder)

    
    # encoder
    hv_flag_encoder = conf.model.encoder_pattern
    if is_rank0():
        logger.info('Converting encoder to MoE')
    convert_feedforward_of_transformer_to_moe(model.encoder.encoder, num_experts, num_experts_per_token, softmax_order,
                                              HV_FLAG='hv' in hv_flag_encoder, pattern=hv_flag_encoder)
    # decoder
    hv_flag_decoder = conf.model.decoder_pattern
    if is_rank0():
        logger.info('Converting decoder to MoE')
    convert_feedforward_of_transformer_to_moe(model.mae_decoder.decoder, num_experts, num_experts_per_token, softmax_order,
                                                HV_FLAG='hv' in hv_flag_decoder, pattern=hv_flag_decoder)

    model.get_router_logits = True


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
        recon_multiplex, recon_he, enc_router_logits, dec_router_logits = outputs
        loss_total, metrics = mae_loss_fn(recon_multiplex, recon_he, multiplex, he)
        zloss = sum(router_z_loss(logits) for logits in enc_router_logits + dec_router_logits) / (len(enc_router_logits) + len(dec_router_logits))
        loss_total += 0.1 * zloss
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                if is_rank0():  
                    logger.info(f'zloss loss @ {self.state.global_step}: {zloss.item():.4f}')
                    wandb.log({'train/zloss': zloss.item()}, step=self.state.global_step)
        return (loss_total, outputs) if return_outputs else loss_total
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # if DISABLE_EVAL:
            # return {}
        # else:
        eval_dataloader = {'test': self.get_eval_dataloader('test')}
        if self.conf.inpainting.track:
            inpaint_test_dataloader = self.get_eval_dataloader('test_inpaint')
            eval_dataloader['test_inpaint'] = inpaint_test_dataloader

        loss_metrics = evaluate_fn(self.conf, self.model, eval_dataloader, self.state.epoch)
        
        return loss_metrics


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

    checkpoints = os.listdir(f'{conf.experiment.dir}/{conf.training.post_training_model_name}/checkpoints')
    checkpoints = list(filter(
        lambda x: 'model.safetensors' in os.listdir(f'{conf.experiment.dir}/{conf.training.post_training_model_name}/checkpoints/{x}'),
        checkpoints
    ))
    checkpoint_time_steps = [int(x.split('-')[1]) for x in checkpoints]
    latest_checkpoint = np.argmax(checkpoint_time_steps)
    checkpoint = checkpoints[latest_checkpoint]
    state_dict = load_checkpoint_safetensors(f'{conf.experiment.dir}/{conf.training.post_training_model_name}/checkpoints/{checkpoint}/model.safetensors')
    mae_model.load_state_dict(state_dict)

    if is_rank0():
        logger.info(f'Loaded model from {checkpoint}')
        logger.info(f'Model has {sum(p.numel() for p in mae_model.parameters() if p.requires_grad)} trainable parameters')
    mae_model.cuda()

    convert_all_modules_to_moe(
        mae_model,
        num_experts=NUM_MOE_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_ACTIVATED,
        softmax_order="softmax_topk"
    )
    if is_rank0():
        logger.info(f'MoE model has {sum(p.numel() for p in mae_model.parameters() if p.requires_grad)} trainable parameters')
        logger.info('Successfully converted model to MoE')


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
        dataloader_num_workers=conf.training.num_workers,
        report_to="wandb",
        run_name=conf.experiment.name,
        gradient_accumulation_steps=conf.training.grad_accumulation,
        learning_rate=conf.training.lr,
        lr_scheduler_type=conf.training.lr_scheduler_type,
        weight_decay=conf.training.weight_decay,
        ddp_find_unused_parameters=True, 
    )

    trainer = CustomTrainer(
        conf=conf, 
        model=mae_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        compute_metrics=compute_metrics, 
        data_collator=custom_collate_fn
    )

    trainer.add_callback(InpaintingUpdateCallback())

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
    for inputs in test_dataloader:

        multiplex, he, multiplex_mask, he_mask, multiplex_channels, _, _ = inputs

        multiplex = to_device(multiplex, 'cuda')
        he = to_device(he, 'cuda')
        multiplex_mask = to_device(multiplex_mask, 'cuda')
        he_mask = to_device(he_mask, 'cuda')
        multiplex_channels = to_device(multiplex_channels, 'cuda')

        # new virtues
        with torch.no_grad():
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
                outputs = model(multiplex, he, multiplex_channels, multiplex_mask=multiplex_mask, he_mask=he_mask)
        recon_multiplex, recon_he, _, _ = outputs
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

    if (hasattr(conf, 'rcp') and conf.rcp) or (hasattr(cli_conf, 'rcp') and cli_conf.rcp):
        # The run is on RCP cluster
        conf.marker_embedding_dir = '/mnt/aimm/scratch/datasets/marker_embeddings/esm2_t30_150M_UR50D'
        conf.experiment.dir = '/mnt/aimm/scratch/shr/mmvirtues/outputs'

    assert hasattr(cli_conf.training, 'post_training_model_name'), "No model specified for post-training"
    assert cli_conf.training.post_training_model_name is not None, "No model specified for post-training"

    ckpt_conf = pickle.load(
        open(f'{conf.experiment.dir}/{cli_conf.training.post_training_model_name}/config.pkl', 'rb')
    )
    conf = OmegaConf.merge(conf, ckpt_conf)

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

    assert hasattr(conf, 'moe'), 'Require MoE parameters for post-training'

    NUM_MOE_EXPERTS = conf.moe.num_experts
    NUM_EXPERTS_ACTIVATED = conf.moe.num_experts_activated
    MOE_FREQ = conf.moe.freq

    conf.experiment.name = f"MoE_num={NUM_MOE_EXPERTS}_freq={MOE_FREQ}_{conf.training.post_training_model_name}"
    if is_rank0():
        logger.info(f'Experiment name: {conf.experiment.name}')


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