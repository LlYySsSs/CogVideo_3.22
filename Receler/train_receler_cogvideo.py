import os
import random
import shutil
import json
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import inspect

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXDDIMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.video_processor import VideoProcessor


from receler.erasers.utils import DisableEraser, save_cogvideo_eraser_to_diffusers_format
from receler.erasers.cogvideo_erasers import setup_cogvideo_adapter_eraser

from receler.concept_reg_cogvideo import get_mask, AttnMapsCapture, EraserOutputsCapture



def get_t5_token_position(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    find_token:str,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    find_token = [find_token]
    batch_size = len(prompt)

    
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids[0]

    token_inputs = tokenizer(
        find_token,
        padding="do_not_pad",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    token_input_ids = token_inputs.input_ids[0]

    p_token = tokenizer.convert_ids_to_tokens(text_input_ids)
    f_token = tokenizer.convert_ids_to_tokens(token_input_ids)
    #print(p_token)

    word_len = len(f_token)
    index = -1
    for i in range(len(p_token)):
        if(p_token[i:i+word_len] == f_token):
            index = i
            break
    return word_len,index



def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


def prepare_latents(
        vae, scheduler, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        vae_scale_factor_spatial = (
            2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
        )
        vae_scale_factor_temporal = (
            vae.config.temporal_compression_ratio if vae is not None else 4
        )
        vae_scaling_factor_image = (
            vae.config.scaling_factor if vae is not None else 0.7
        )

        shape = (
            batch_size,
            (num_frames - 1) // vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
        return latents

def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        vae_scaling_factor_image = (
            vae.config.scaling_factor if vae is not None else 0.7
        )
        latents = 1 / vae_scaling_factor_image * latents

        frames = vae.decode(latents).sample
        return frames


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def prepare_extra_step_kwargs(scheduler, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

@torch.no_grad()
def sample_till_t(latents,prompt_embeds,negative_prompt_embeds,image_rotary_emb,timesteps,stop_t, transformer,scheduler,dtype,extra_step_kwargs):
    # generate an image with the concept from model
    old_pred_original_sample = None
    guidance_scale = 7.0
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    for i, t in enumerate(timesteps):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])

        # predict noise model_output
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if not isinstance(scheduler, CogVideoXDPMScheduler):
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )
        latents = latents.to(dtype)
        if stop_t is not None and i == stop_t:
            break

    return latents


def apply_model(noisy_model_input,prompt_embeds,timesteps,image_rotary_emb,transformer,scheduler):
    model_output = transformer(
        hidden_states=noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps,
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
    )[0]
    model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)
    return model_pred


def train_receler(
        args,
        concept,
        save_root,
        iterations=1000,
        lr=3e-4,
        start_guidance=3,
        negative_guidance=1,
        seperator=None,
        height = 480,
        width = 720,
        num_frames = 4,
        inference_steps = 50,
    ):

    #setup device and pipe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32


    # extend specific concept
    word_print = concept.replace(' ', '')
    original_concept = concept

    concept_mappings = {
        'i2p': "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood",
    }

    concept = concept_mappings.get(original_concept, concept)

    # seperate concept string into (multiple) concepts
    if seperator is not None:
        words = concept.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [concept]

    # load model

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=None,
        variant=None,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
    )

    vae_scale_factor_spatial = (
            2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    )
    scheduler = CogVideoXDDIMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    

    # We only train the additional adapter layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    text_encoder.to(device, dtype=dtype)
    transformer.to(device, dtype=dtype)
    vae.to(device, dtype=dtype)

    # setup eraser
    erasers = setup_cogvideo_adapter_eraser(transformer, eraser_rank=args.eraser_rank, device=device, dtype = dtype)

    # setup optimizer
    opt = torch.optim.Adam([param for eraser in erasers.values() for param in eraser.parameters()], lr=lr)
    

    # setup experiment name
    erase_msg = f'rank_{args.eraser_rank}'
    #advrs_msg = f'advrs_iter_{args.advrs_iters}-start_{args.start_advrs_train}-num_prompts_{args.num_advrs_prompts}'
    reg_msg = f'concept_reg_{args.concept_reg_weight}-mask_thres_{args.mask_thres}'
    name = f'receler-word_{word_print}-{erase_msg}-{reg_msg}-iter_{iterations}-lr_{lr}'

    print('\n'.join(['#'*50, name, '#'*50]))

    folder_path = os.path.join(save_root, name)
    
    # dicts to store captured attention maps and eraser outputs
    attn_maps = {}
    eraser_outs = {}

    

    # training
    pbar = tqdm(range(iterations))
    for it in pbar:

        transformer.train()
        
        #word_idx, word = random.sample(list(enumerate(words)),1)[0]
        word = "beagle dog"

        # get text embeddings for unconditional and conditional prompts
        word_len,token_idx = get_t5_token_position(tokenizer=tokenizer,text_encoder=text_encoder,prompt=word,find_token=word)
        emb_0 = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                "",
                model_config.max_text_seq_length,
                device,
                dtype,
                requires_grad=False,
        )
        #classifier-free guidance + negative guidance
        #conditional： "dog" + dog的图片 “cat” + cat的图片
        #unconditional: "" + dog cat的图片
        #inference:
        #"dog" -> pred_cond
        #"" -> pred_uncond
        #\alpha * pred_cond - \beta * pred_uncond
        emb_p = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                "{}".format(word),
                model_config.max_text_seq_length,
                device,
                dtype,
                requires_grad=False,
        )
        emb_n = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                "{}".format(word),
                model_config.max_text_seq_length,
                device,
                dtype,
                requires_grad=False,
        )

        # hacking the indices of targeted word and adversarial prompts
        word_indices = torch.arange(1, word_len, device=device)
        advrs_indices = torch.arange(1 + word_len, 1 + word_len + args.num_advrs_prompts, device=device)

        # get timesteps for v-prediction
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, inference_steps, device)
        _num_timesteps = len(timesteps)

        generator=torch.Generator()
        #prepare latents
        latent_channels = transformer.config.in_channels
        
        start_code = prepare_latents(
            vae,
            scheduler,
            1,
            latent_channels,
            num_frames,
            height,
            width,
            dtype,
            transformer.device,
            generator=generator
        )
        #print("start_code:",start_code.shape)

        patch_height = start_code.shape[-2] // transformer.patch_embed.patch_size
        patch_width = start_code.shape[-1] // transformer.patch_embed.patch_size

        image_rotary_emb = (None)

        
        extra_step_kwargs = prepare_extra_step_kwargs(scheduler,generator, 0.0)
        
        stop_t = random.randint(0,_num_timesteps-2)
                
        z = sample_till_t(start_code,emb_p,emb_0,image_rotary_emb,timesteps,stop_t,transformer,scheduler,dtype,extra_step_kwargs)
        #batchsize,F,num_channels,height,width
        # video = decode_latents(vae,z)
        # video = CogVideoXPipelineOutput(frames=video_processor.postprocess_video(video=video, output_type="pil"))
        # export_to_video(video.frames[0],f'{it}.mp4',fps=8)
        # break   
        #the next step
        t = timesteps[stop_t + 1].unsqueeze(0)
            
        with torch.no_grad():
            # get conditional and unconditional scores from frozen model at time step t and image z
            with DisableEraser(transformer, train=False):
                #e_0 = model.apply_model(z.to(device), t_enc_ddpm.to(device), emb_0.to(device))
                v_0 = apply_model(z,emb_0,t,image_rotary_emb,transformer,scheduler)
                #print("v_0:",v_0.shape)
                with AttnMapsCapture(transformer, attn_maps=attn_maps):
                    #e_p = model.apply_model(z.to(device), t_enc_ddpm.to(device), emb_p.to(device))
                    v_p = apply_model(z,emb_p,t,image_rotary_emb,transformer,scheduler)
                    #print("v_p:",v_p.shape)
        
        

        v_0.requires_grad = False
        v_p.requires_grad = False

        #get attn_masks
        attn_masks = get_mask(attn_maps, word_indices, args.mask_thres, patch_height, patch_width)
        #print('attn_masks["transformer_blocks.0.attn1.attn"].shape = ',attn_masks["transformer_blocks.0.attn1.attn"].shape)
        #get_mask(attn_maps, word_indices, args.mask_thres)

        # for inner_it in range(args.advrs_iters):

        #     # copy advrs_prompt_emb to input emb_n and make it requires_grad if advrs train
        #     emb_n = emb_n.detach()
        #     emb_n[:, advrs_indices, :].data = advrs_prompt_embs[word_idx].data
        #     emb_n.requires_grad = True

        # get conditional score from model
        with EraserOutputsCapture(transformer, erasers, eraser_outs):
            v_n = apply_model(z, emb_n, t, image_rotary_emb, transformer, scheduler)
            #print("v_n:",v_n.shape)
        
        


        
        loss_total = torch.tensor(0.).to(device)
        # v_0.requires_grad = False
        # v_p.requires_grad = False

        loss_erase = F.mse_loss(v_n, v_0 - (negative_guidance * (v_p - v_0)))
        loss_total += loss_erase
        # compute cross attn regularization loss
        loss_eraser_reg = torch.tensor(0.).to(device)
        reg_count = 0
        for e_name, e_out in eraser_outs.items():
            mask_name = ".".join(e_name.split(".")[:-1]) + ".attn"
            if mask_name not in attn_masks:
                print(f'Warning: cannot compute regularization loss for {mask_name}, because corresponding mask not found.')  # cannot find mask for regularizing
                continue
            reg_count += 1
            mask = attn_masks[mask_name]
            flip_mask = (~mask.bool()).float().unsqueeze(-1)  #B, F, height, width, 1
            #print(flip_mask.sum())
            
            e_reshape = e_out.view(flip_mask.shape[0],flip_mask.shape[1],flip_mask.shape[2],flip_mask.shape[3],-1)
            #B,F,height,width,dim
            
            loss_eraser_reg += ((e_reshape * flip_mask) ** 2).mean(-1).sum() / (flip_mask.sum() + 1e-9)
        loss_eraser_reg /= reg_count
        loss_total += args.concept_reg_weight * loss_eraser_reg

        # update weights to erase the concept
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        pbar.set_postfix({"loss_total": loss_total.item()}, refresh=False)
        pbar.set_description_str(f"[{datetime.now().strftime('%H:%M:%S')}] Erase", refresh=False)

        # visualization
        if it == 0 or (it+1) % args.visualize_iters == 0:
            folder = os.path.join(os.path.join(save_root, f'{name}', 'visualize', f'iter_{it}'))
            os.makedirs(folder, exist_ok=True)
            transformer.eval()
            with torch.no_grad():
                vis_code = prepare_latents(
                    vae,
                    scheduler,
                    1,
                    latent_channels,
                    num_frames,
                    480,
                    720,
                    dtype,
                    transformer.device,
                    generator=torch.Generator().manual_seed(711)
                )
                #改这里
                timesteps, num_inference_steps = retrieve_timesteps(scheduler, inference_steps, device)
                _num_timesteps = len(timesteps)

                video = sample_till_t(vis_code,emb_p,emb_0, image_rotary_emb,timesteps,None,transformer,scheduler,dtype,extra_step_kwargs)
                video = decode_latents(vae,video)
                video = CogVideoXPipelineOutput(frames=video_processor.postprocess_video(video=video, output_type="pil"))
                export_to_video(video.frames[0],os.path.join(folder, f'{it}.mp4'),fps=8)

        # save checkpoint
        if (it+1) % args.save_ckpt_iters == 0:
            transformer.eval()
            save_cogvideo_eraser_to_diffusers_format(
                os.path.join(folder_path, 'ckpts', f'iter_{it}'),
                erasers=erasers,
                eraser_rank=args.eraser_rank,
            )

    transformer.eval()
    save_cogvideo_eraser_to_diffusers_format(
        folder_path,  # save last ckpt directly under folder_path
        erasers=erasers,
        eraser_rank=args.eraser_rank,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('--concept', help='Concept to erase. Multiple concepts can be separated by commas.', type=str, required=True)
    parser.add_argument('--save_root', help='root to save model checkpoint', type=str, required=False, default='./models/')
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=200)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-4)

    # Arguments for cross-attention regularization
    parser.add_argument('--concept_reg_weight', help='weight of concept-localized regularization loss', type=float, default=0.1)
    parser.add_argument('--mask_thres', help='threshold to obtain cross-attention mask', type=float, default=0.1)

    # Arguments for adversarial training
    parser.add_argument('--advrs_iters', help='number of adversarial iterations', type=int, default=50)
    parser.add_argument('--start_advrs_train', help='iteration to start adversarial training', type=int, default=0)
    parser.add_argument('--num_advrs_prompts', help='number of attack prompts to add', type=int, default=16)

    # Save checkpoint and visualization arguments
    parser.add_argument('--save_ckpt_iters', help="save checkpoint every N iterations", type=int, default=20)
    parser.add_argument('--visualize_iters', help="generate images every N iterations", type=int, default=20)
    parser.add_argument('--num_visualize', help="number of images to visualize", type=int, default=3)

    # Other training configuration
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1.0)
    parser.add_argument('--eraser_rank', help='the rank of eraser', type=int, required=False, default=128)
    parser.add_argument('--pretrained_model_name_or_path', help='path for pretrained model', type=str, required=False, default='/data1/yexiaoyu/cogvideox-2b')
    #parser.add_argument('--pretrained_cfg', help='config path for stable diffusion v1-4', type=str, required=False, default='./receler/configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--height', help='height used to train', type=int, required=False, default=480)
    parser.add_argument('--width', help='width used to train', type=int, required=False, default=720)
    parser.add_argument('--num_frames', help='number of frames used to train', type=int, required=False, default=9)#5
    parser.add_argument('--time_steps', help='time steps of inference used to train', type=int, required=False, default=50)

    args = parser.parse_args()

    train_receler(
        args,
        args.concept,
        args.save_root,
        iterations=args.iterations,
        lr=args.lr,
        start_guidance=args.start_guidance,
        negative_guidance=args.negative_guidance,
        seperator=args.seperator,
        height = args.height,
        width = args.width,
        num_frames= args.num_frames,
        inference_steps=args.time_steps,
    )
