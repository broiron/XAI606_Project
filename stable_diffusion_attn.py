from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel,logging,CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import StableDiffusionPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from crs_controller import *

import time
MY_TOKEN = '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'

class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4',concept_name=None, latent_mode=True):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=MY_TOKEN, torch_dtype=torch.float16).to(device)
        self.device = device
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        logger.info(f'loading stable diffusion with {model_name}...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        # self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)
        self.vae = self.pipeline.vae

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder

        self.image_encoder = None
        self.image_processor = None


        # 3. The UNet model for generating the latents.
        # self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(self.device)
        self.unet = self.pipeline.unet

        # 4. Create a scheduler for inference
        # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.scheduler = self.pipeline.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        if concept_name is not None:
            self.load_concept(concept_name)
        logger.info(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name):
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def delta_denoising_score(self, controller, tgt_emb, ref_emb, z, z_hat, guidance_scale=100, min_step=20, max_step=980, prompt=None, timestep=None,what_mask=None):
        latents = z
        ref_latents = z_hat
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep is None:
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = timestep
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            ref_latents_noisy = self.scheduler.add_noise(ref_latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            ref_latent_model_input = torch.cat([ref_latents_noisy] * 2)
            tgt_emb_list = []
            ref_emb_list = []
            for i in range(latent_model_input.shape[0]//2):
                tgt_emb_list.append(tgt_emb)
                ref_emb_list.append(ref_emb)
            tgt_emb = torch.cat(tgt_emb_list, dim=0)
            ref_emb = torch.cat(ref_emb_list, dim=0)
            # tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb).sample
            tgt_dtype = latent_model_input.dtype
            latent_model_input = latent_model_input.to(tgt_emb.dtype)
            tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb)['sample']
            if what_mask == 'tgt':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
            # for i in range(len(img)):
            #     np_img = img[i]
            #     image = Image.fromarray(np_img)
            #     image.save(f"./crsattn_img_{i}.png")
            ref_dtype = ref_latent_model_input.dtype
            ref_latent_model_input = ref_latent_model_input.to(ref_emb.dtype)
            ref_noise_pred = self.unet(ref_latent_model_input, t, encoder_hidden_states=ref_emb)['sample']
            if what_mask == 'ref':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        tgt_noise_pred_uncond, tgt_noise_pred_text = tgt_noise_pred.chunk(2)
        tgt_noise_pred = tgt_noise_pred_uncond + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)
        latents = controller.step_callback(latents)

        # perform guidance (high scale from paper!)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        tgt_grad = w * (tgt_noise_pred - noise)
        ref_grad = w * (ref_noise_pred - noise)

        dds = tgt_grad - ref_grad

        return dds, t, tgt_grad, ref_grad, img

    def get_score(self, controller, ref_emb, z, guidance_scale=100, min_step=20, max_step=980, prompt=None, timestep=None,show_attn=True):
        latents = z
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep is None:
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = timestep

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = latent_model_input.to(ref_emb.dtype)

            ref_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=ref_emb)['sample']
            if show_attn:
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')


        # perform guidance (high scale from paper!)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)
        latents = controller.step_callback(latents)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        sds = w * (ref_noise_pred - noise)

        return sds, t, img

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''], padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, inputs, guidance_scale=100, mask=None, sds_weight=None):
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * sds_weight
        grad = grad * mask.to(grad.device)
        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0, grad

    def predict_noise(self, text_embeddings, inputs, guidance_scale=100):
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)
        grad = (noise_pred - noise)

        return noise_pred, noise, w, grad


    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def aggregate_attention(self, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt=None):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()


    def show_cross_attention(self, attention_store: AttentionStore, res: int, from_where: List[str], prompt=None, select: int = 0):
        prompt = [prompt]
        tokens = self.tokenizer.encode(prompt[select])
        # print('a prompt is',prompt)
        decoder = self.tokenizer.decode
        attention_maps = self.aggregate_attention(attention_store, res, from_where, True, select, prompt=prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            images.append(image)

        return images
        

    def show_self_attention_comp(self, attention_store: AttentionStore, res: int, from_where: List[str],
                            max_com=10, select: int = 0):
        attention_maps = self.aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        return images

class StableDiffusionControlnet(nn.Module):
    # def __init__(self, device, model_name='stabilityai/stable-diffusion-2-1',concept_name=None, latent_mode=True): 
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4',concept_name=None, latent_mode=True):
    # def __init__(self, device, model_name='runwayml/stable-diffusion-v1-5',concept_name=None, latent_mode=True):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload("1")
        # self.pipeline = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=MY_TOKEN, torch_dtype=torch.float16).to(device)
        self.device = device
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.2)
        self.max_step = int(self.num_train_timesteps * 0.6)

        logger.info(f'loading stable diffusion with {model_name}...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        # self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)
        self.vae = self.pipeline.vae

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder

        self.image_encoder = None
        self.image_processor = None


        # 3. The UNet model for generating the latents.
        # self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(self.device)
        self.unet = self.pipeline.unet

        # 4. Create a scheduler for inference
        # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.scheduler = self.pipeline.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        if concept_name is not None:
            self.load_concept(concept_name)
        logger.info(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name):
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def delta_denoising_score(self, controller, depth_map, tgt_emb, ref_emb, z, z_hat, guidance_scale=100, min_step=500, max_step=980, prompt=None, timestep=None,what_mask=None):
        latents = z
        ref_latents = z_hat
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep is None:
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = timestep
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            ref_latents_noisy = self.scheduler.add_noise(ref_latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2).to(self.controlnet.dtype)
            ref_latent_model_input = torch.cat([ref_latents_noisy] * 2).to(self.controlnet.dtype)
            depth_map = depth_map.to(self.controlnet.dtype)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_emb,
                controlnet_cond=depth_map,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            ref_down_block_res_samples, ref_mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_emb,
                controlnet_cond=depth_map,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            latent_model_input = latent_model_input.to(tgt_emb.dtype)
            # tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb).sample
            tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
            if what_mask == 'tgt':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
            # for i in range(len(img)):
            #     np_img = img[i]
            #     image = Image.fromarray(np_img)
            #     image.save(f"./crsattn_img_{i}.png")
            ref_latent_model_input = ref_latent_model_input.to(ref_emb.dtype)
            ref_noise_pred = self.unet(ref_latent_model_input, t, encoder_hidden_states=ref_emb,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=ref_down_block_res_samples,
                    mid_block_additional_residual=ref_mid_block_res_sample,
                    return_dict=False,
                )[0]
            if what_mask == 'ref':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        tgt_noise_pred_uncond, tgt_noise_pred_text = tgt_noise_pred.chunk(2)
        tgt_noise_pred = tgt_noise_pred_uncond + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

        # perform guidance (high scale from paper!)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        tgt_grad = w * (tgt_noise_pred - noise)
        ref_grad = w * (ref_noise_pred - noise)

        dds = tgt_grad - ref_grad

        return dds, t, tgt_grad, ref_grad, img

    def delta_denoising_score_attn(self, depth_map, tgt_emb, ref_emb, z, z_hat, guidance_scale=100, min_step=500, max_step=980, prompt=None, timestep=None, controller=None, what_mask=None):
        latents = z
        ref_latents = z_hat
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if timestep is None:
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = timestep
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            ref_latents_noisy = self.scheduler.add_noise(ref_latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            ref_latent_model_input = torch.cat([ref_latents_noisy] * 2)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_emb,
                controlnet_cond=depth_map,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            ref_down_block_res_samples, ref_mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_emb,
                controlnet_cond=depth_map,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )

            
            # tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb).sample
            tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

            if what_mask == 'tgt':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()
            # for i in range(len(img)):
            #     np_img = img[i]
            #     image = Image.fromarray(np_img)
            #     image.save(f"./crsattn_img_{i}.png")
            ref_noise_pred = self.unet(ref_latent_model_input, t, encoder_hidden_states=ref_emb,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=ref_down_block_res_samples,
                    mid_block_additional_residual=ref_mid_block_res_sample,
                    return_dict=False,
                )[0]
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')
            
            if what_mask == 'ref':
                img = self.show_cross_attention(controller, 16, ("up", "down"), prompt=prompt)
            controller.reset()

        # perform guidance (high scale from paper!)
        tgt_noise_pred_uncond, tgt_noise_pred_text = tgt_noise_pred.chunk(2)
        tgt_noise_pred = tgt_noise_pred_uncond + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

        # perform guidance (high scale from paper!)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        tgt_grad = w * (tgt_noise_pred - noise)
        ref_grad = w * (ref_noise_pred - noise)

        dds = tgt_grad - ref_grad

        return dds, t, tgt_grad, ref_grad, img

    def delta_denoising_score_anneal(self, controller, tgt_emb, ref_emb, z, z_hat, guidance_scale=100, max_step=980):
        latents = z
        ref_latents = z_hat
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            ref_latents_noisy = self.scheduler.add_noise(ref_latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            ref_latent_model_input = torch.cat([ref_latents_noisy] * 2)

            tgt_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=tgt_emb).sample

            ref_noise_pred = self.unet(ref_latent_model_input, t, encoder_hidden_states=ref_emb).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        tgt_noise_pred_uncond, tgt_noise_pred_text = tgt_noise_pred.chunk(2)
        tgt_noise_pred = tgt_noise_pred_uncond + guidance_scale * (tgt_noise_pred_text - tgt_noise_pred_uncond)

        latents = controller.step_callback(latents)
        # perform guidance (high scale from paper!)
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = ref_noise_pred_uncond + guidance_scale * (ref_noise_pred_text - ref_noise_pred_uncond)

        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        tgt_grad = w * (tgt_noise_pred - noise)
        ref_grad = w * (ref_noise_pred - noise)

        dds = tgt_grad - ref_grad

        return dds, t

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''], padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, inputs, guidance_scale=100, mask=None, sds_weight=None):
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * sds_weight
        grad = grad * mask.to(grad.device)
        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0, grad

    def predict_noise(self, text_embeddings, inputs, guidance_scale=100):
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = inputs
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        # grad = w * (noise_pred - noise)
        grad = (noise_pred - noise)

        return noise_pred, noise, w, grad


    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

    def aggregate_attention(self, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompt=None):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompt), -1, res, res, item.shape[-1])[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()


    def show_cross_attention(self, attention_store: AttentionStore, res: int, from_where: List[str], prompt=None, select: int = 0):
        prompt = [prompt]
        tokens = self.tokenizer.encode(prompt[select])
        # print('a prompt is',prompt)
        decoder = self.tokenizer.decode
        attention_maps = self.aggregate_attention(attention_store, res, from_where, True, select, prompt=prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            images.append(image)

        return images
        

    def show_self_attention_comp(self, attention_store: AttentionStore, res: int, from_where: List[str],
                            max_com=10, select: int = 0):
        attention_maps = self.aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image)
            images.append(image)
        return images


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()



