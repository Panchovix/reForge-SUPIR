import os
import gradio as gr
import ldm_patched.modules.model_management as mm
import ldm_patched.utils.path_utils as folder_paths
from SUPIR.util import load_state_dict, convert_dtype
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from modules import scripts
from SUPIR.util import PIL2Tensor, Tensor2PIL
from ldm_patched.contrib.external import ImageScaleBy
class SUPIRScript:
    sorting_priority = 15  # Adjust this as needed

    def title(self):
        return "SUPIR Upscaler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Tab("SUPIR Upscale"):
                supir_model = gr.Dropdown(label="SUPIR Model", choices=folder_paths.get_filename_list("checkpoints"), value="")
                sdxl_model = gr.Dropdown(label="SDXL Model", choices=folder_paths.get_filename_list("checkpoints"), value="")
                image = gr.Image(label="Input Image", type="numpy")
                seed = gr.Slider(label="Seed", minimum=0, maximum=0xffffffffffffffff, step=1, value=123)
                
                resize_method = gr.Dropdown(label="Resize Method", choices=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], value="lanczos")
                scale_by = gr.Number(label="Scale By", value=1.0, step=0.01, precision=2)
                steps = gr.Slider(label="Steps", minimum=3, maximum=4096, step=1, value=45)
                restoration_scale = gr.Number(label="Restoration Scale", value=-1.0, step=1.0)
                
                cfg_scale = gr.Number(label="CFG Scale", value=4.0, step=0.01)
                a_prompt = gr.Textbox(label="Positive Prompt", value="high quality, detailed", interactive=True)
                n_prompt = gr.Textbox(label="Negative Prompt", value="bad quality, blurry, messy", interactive=True)
                
                s_churn = gr.Slider(label="EDM s Churn", minimum=0, maximum=40, step=1, value=5)
                s_noise = gr.Number(label="s Noise", value=1.003, step=0.001)
                control_scale = gr.Number(label="Control Scale", value=1.0, step=0.05)
                cfg_scale_start = gr.Number(label="CFG Scale Start", value=4.0, step=0.05)
                control_scale_start = gr.Number(label="Control Scale Start", value=0.0, step=0.05)

                color_fix_type = gr.Radio(label="Color Fix Type", choices=['None', 'AdaIn', 'Wavelet'], value='Wavelet')
                use_tiled_vae = gr.Checkbox(label="Use Tiled VAE", value=True)
                encoder_tile_size = gr.Slider(label="Encoder Tile Size", minimum=64, maximum=8192, step=64, value=512)
                decoder_tile_size = gr.Slider(label="Decoder Tile Size", minimum=32, maximum=8192, step=64, value=64)

                execute_button = gr.Button("Execute")

            # Define the action on button click
            def run_model(supir_model, sdxl_model, image, seed, resize_method, scale_by, steps, restoration_scale,
                          cfg_scale, a_prompt, n_prompt, s_churn, s_noise, control_scale, 
                          cfg_scale_start, control_scale_start, color_fix_type, use_tiled_vae,
                          encoder_tile_size, decoder_tile_size):
                # Model loading and running logic
                device = mm.get_torch_device()
                mm.unload_all_models()

                SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)
                SDXL_MODEL_PATH = folder_paths.get_full_path("checkpoints", sdxl_model)

                config_path = os.path.join(folder_paths.get_script_directory(), "options/SUPIR_v0.yaml")
                config = OmegaConf.load(config_path)

                model = instantiate_from_config(config.model).cpu()

                # Load the state dicts
                supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
                sdxl_state_dict = load_state_dict(SDXL_MODEL_PATH)

                model.load_state_dict(supir_state_dict, strict=False)
                model.load_state_dict(sdxl_state_dict, strict=False)

                if use_tiled_vae:
                    model.init_tile_vae(encoder_tile_size=encoder_tile_size, decoder_tile_size=decoder_tile_size)

                # Image processing
                image_tensor, orig_height, orig_width = PIL2Tensor(image)
                upscaled_image, = ImageScaleBy.upscale(self, image_tensor, resize_method, scale_by)

                # Sampling details (using the model's sampling method)
                samples = model.batchify_sample(upscaled_image, [a_prompt], num_steps=steps, 
                                                 restoration_scale=restoration_scale, s_churn=s_churn,
                                                 s_noise=s_noise, cfg_scale=cfg_scale, control_scale=control_scale,
                                                 seed=seed, p_p=a_prompt, n_p=n_prompt, color_fix_type=color_fix_type)
                
                # Convert the output sample back to an image format
                final_image = Tensor2PIL(samples)
                
                return final_image

            execute_button.click(fn=run_model, inputs=[supir_model, sdxl_model, image, seed, resize_method, 
                                                         scale_by, steps, restoration_scale, cfg_scale, 
                                                         a_prompt, n_prompt, s_churn, s_noise, control_scale,
                                                         cfg_scale_start, control_scale_start, color_fix_type,
                                                         use_tiled_vae, encoder_tile_size, decoder_tile_size],
                                 outputs=image)

        return (supir_model, sdxl_model, image, seed, resize_method, scale_by, steps, restoration_scale,
                cfg_scale, a_prompt, n_prompt, s_churn, s_noise, control_scale, cfg_scale_start, 
                control_scale_start, color_fix_type, use_tiled_vae, encoder_tile_size, decoder_tile_size)
