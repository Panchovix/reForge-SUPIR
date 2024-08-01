import torch
from sgm.models.diffusion import DiffusionEngine
from sgm.util import instantiate_from_config
import copy
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
import random
from SUPIR.utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from pytorch_lightning import seed_everything
from SUPIR.utils.tilevae import VAEHook
from SUPIR.util import convert_dtype
from contextlib import nullcontext
import ldm_patched.modules.model_management

device = ldm_patched.modules.model_management.get_torch_device()

class SUPIRModel(DiffusionEngine):
    def __init__(self, control_stage_config, ae_dtype='fp32', diffusion_dtype='fp32', p_p='', n_p='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        control_model = instantiate_from_config(control_stage_config)
        self.model.load_control_model(control_model)
        self.first_stage_model.denoise_encoder = copy.deepcopy(self.first_stage_model.encoder)
        self.sampler_config = kwargs['sampler_config']

        self.ae_dtype = convert_dtype(ae_dtype)
        self.model.dtype = convert_dtype(diffusion_dtype)

        self.p_p = p_p
        self.n_p = n_p

    @torch.no_grad()
    def encode_first_stage(self, x):
        #with torch.autocast(device, dtype=self.ae_dtype):
        autocast_condition = (self.ae_dtype == torch.float16 or self.ae_dtype == torch.bfloat16) and not ldm_patched.modules.model_management.is_device_mps(device)
        with torch.autocast(ldm_patched.modules.model_management.get_autocast_device(device), dtype=self.ae_dtype) if autocast_condition else nullcontext():
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        #with torch.autocast(device, dtype=self.ae_dtype):
        self.first_stage_model.to(self.ae_dtype)
        autocast_condition = (self.model.dtype == torch.float16 or self.model.dtype == torch.bfloat16) and not ldm_patched.modules.model_management.is_device_mps(device)
        with torch.autocast(ldm_patched.modules.model_management.get_autocast_device(device), dtype=self.ae_dtype) if autocast_condition else nullcontext():
            if is_stage1:
                h = self.first_stage_model.denoise_encoder_s1(x)
            else:
                h = self.first_stage_model.denoise_encoder(x)
            moments = self.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            if use_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        autocast_condition = (self.ae_dtype == torch.float16 or self.ae_dtype == torch.bfloat16) and not ldm_patched.modules.model_management.is_device_mps(device)
        with torch.autocast(ldm_patched.modules.model_management.get_autocast_device(device), dtype=self.ae_dtype) if autocast_condition else nullcontext():
            out = self.first_stage_model.decode(z)
        return out.float()

    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        '''
        [N, C, H, W], [-1, 1], RGB
        '''
        x = self.encode_first_stage_with_denoise(x, use_sample=False, is_stage1=is_stage1)
        return self.decode_first_stage(x)

    @torch.no_grad()
    def batchify_sample(self, x, p, p_p='default', n_p='default', num_steps=100, restoration_scale=4.0, s_churn=0, s_noise=1.003, cfg_scale=4.0, seed=-1,
                        num_samples=1, control_scale=1, color_fix_type='None', use_linear_CFG=False, use_linear_control_scale=False,
                        cfg_scale_start=1.0, control_scale_start=0.0, **kwargs):
        '''
        [N, C], [-1, 1], RGB
        '''
        assert len(x) == len(p)
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        N = len(x)
        if num_samples > 1:
            assert N == 1
            N = num_samples
            x = x.repeat(N, 1, 1, 1)
            p = p * N

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p

        self.sampler_config.params.num_steps = num_steps
        if use_linear_CFG:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale_start
        else:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale
        self.sampler_config.params.restore_cfg = restoration_scale
        self.sampler_config.params.s_churn = s_churn
        self.sampler_config.params.s_noise = s_noise
        self.sampler = instantiate_from_config(self.sampler_config)

        print("Sampler: ", self.sampler_config.target)
        print("sampler_config: ", self.sampler_config.params)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        
        self.model.to('cpu')
        self.conditioner.to('cpu')

        # stage 1: encode/decode/encode
        self.first_stage_model.to(device)
        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)
        self.first_stage_model.to('cpu')

        #conditioning
        self.conditioner.to(device)
        c, uc = self.prepare_condition(_z, p, p_p, n_p, N)
        self.conditioner.to('cpu')

        denoiser = lambda input, sigma, c, control_scale: self.denoiser(
            self.model, input, sigma, c, control_scale, **kwargs
        )
        noised_z = torch.randn_like(_z).to(_z.device)
        
        ldm_patched.modules.model_management.soft_empty_cache()

        #sampling
        self.model.diffusion_model.to(device)
        self.model.control_model.to(device)
        self.denoiser.to(device)
        
        _samples = self.sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=control_scale,
                                use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)
        self.model.diffusion_model.to('cpu')
        self.model.control_model.to('cpu')
        
        #decoding
        self.first_stage_model.to(device)
        samples = self.decode_first_stage(_samples)
        self.first_stage_model.to('cpu')
        
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)
        return samples

    def init_tile_vae(self, encoder_tile_size=512, decoder_tile_size=64):
        self.first_stage_model.denoise_encoder.original_forward = self.first_stage_model.denoise_encoder.forward
        self.first_stage_model.encoder.original_forward = self.first_stage_model.encoder.forward
        self.first_stage_model.decoder.original_forward = self.first_stage_model.decoder.forward
        self.first_stage_model.denoise_encoder.forward = VAEHook(
            self.first_stage_model.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.encoder.forward = VAEHook(
            self.first_stage_model.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.decoder.forward = VAEHook(
            self.first_stage_model.decoder, decoder_tile_size, is_decoder=True, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        
    def prepare_condition(self, _z, p, p_p, n_p, N):
        batch = {}
        batch['original_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['crop_coords_top_left'] = torch.tensor([0, 0]).repeat(N, 1).to(_z.device)
        batch['target_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['aesthetic_score'] = torch.tensor([9.0]).repeat(N, 1).to(_z.device)
        batch['control'] = _z

        batch_uc = copy.deepcopy(batch)
        batch_uc['txt'] = [n_p for _ in p]
        autocast_condition = (self.model.dtype == torch.float16 or self.model.dtype == torch.bfloat16) and not ldm_patched.modules.model_management.is_device_mps(device)
        if not isinstance(p[0], list):
            print("Using local prompt: ")
            batch['txt'] = [''.join([_p, p_p]) for _p in p]
            print(batch['txt'])
            with torch.autocast(ldm_patched.modules.model_management.get_autocast_device(device), dtype=self.model.dtype) if autocast_condition else nullcontext():
                c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        else:
            print("Using tile prompts")
            assert len(p) == 1, 'Support bs=1 only for local prompt conditioning.'
            p_tiles = p[0]
            c = []
            for i, p_tile in enumerate(p_tiles):
                batch['txt'] = [''.join([p_tile, p_p])]
                with torch.autocast(ldm_patched.modules.model_management.get_autocast_device(device), dtype=self.model.dtype) if autocast_condition else nullcontext():
                    if i == 0:
                        _c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
                    else:
                        _c, _ = self.conditioner.get_unconditional_conditioning(batch, None)
                c.append(_c)
        return c, uc