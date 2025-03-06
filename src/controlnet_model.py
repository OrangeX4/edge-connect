import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from diffusers import ControlNetModel, UniPCMultistepScheduler
from .pipeline_stable_diffusion_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline


class ControlNetInpaintingModel(nn.Module):
    def __init__(self, config=None):
        super(ControlNetInpaintingModel, self).__init__()
        self.config = config
        
        # Load pre-trained ControlNet model and pipeline
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", 
            torch_dtype=torch.float16
        )
        
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        )
        
        # Use faster scheduler
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        
        # Enable memory efficient attention if xformers is available
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass
            
        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipeline.to('cuda')
    
    def forward(self, images, edges, masks, prompt="", num_inference_steps=20, 
                controlnet_conditioning_scale=0.5, guidance_scale=7.5, seed=None):
        """
        Forward pass for ControlNet inpainting.
        
        Args:
            images: Input image to be inpainted (PIL Image, numpy array or PyTorch tensor)
            edges: Edge map for control (PIL Image, numpy array or PyTorch tensor)
            masks: Mask indicating regions to be inpainted (PIL Image, numpy array or PyTorch tensor)
            prompt: Text prompt for guiding the inpainting
            num_inference_steps: Number of denoising steps
            controlnet_conditioning_scale: Strength of controlnet guidance
            guidance_scale: Strength of text guidance
            seed: Random seed for reproducibility
            
        Returns:
            Inpainted image as a tensor matching the input format
        """
        # Get original device and batch info
        device = None
        if isinstance(images, torch.Tensor):
            device = images.device
            batch_size = images.shape[0] if images.ndim == 4 else 1
        
        # Convert tensors/numpy arrays to PIL images
        def convert_to_pil(img, is_edge=False):
            if isinstance(img, Image.Image):
                return img
            elif isinstance(img, torch.Tensor):
                # Handle tensor input - convert to numpy
                if img.ndim == 4:  # batch dimension
                    img = img[0]  # Take first image in batch
                img = img.detach().cpu().numpy()
                
                # Special handling for edge maps or 1-channel images
                if is_edge or (len(img.shape) == 3 and img.shape[0] == 1) or len(img.shape) < 3:
                    # Handle single channel data (edges, masks)
                    if len(img.shape) == 3 and img.shape[0] == 1:  # CHW format with 1 channel
                        img = img[0]  # Remove channel dimension
                    
                    # Make sure we have a 2D array
                    if len(img.shape) > 2:
                        # Handle unusual shapes like (1, 1, 256)
                        img = np.squeeze(img)
                    
                    # For any 1D array, reshape it to 2D
                    if len(img.shape) == 1:
                        side = int(np.sqrt(img.shape[0]))
                        img = img.reshape(side, side)
                    
                    # Ensure proper value range for images
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Convert grayscale to RGB if needed
                    return Image.fromarray(img).convert('RGB')
                else:
                    # Standard RGB image processing
                    # Convert from CHW to HWC if needed
                    if img.shape[0] == 3 and len(img.shape) == 3:
                        img = img.transpose(1, 2, 0)
                    # Ensure proper value range for images
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    return Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                # Handle numpy array input
                # Special handling for edge maps or 1-channel images
                if is_edge or (len(img.shape) == 3 and img.shape[0] == 1) or len(img.shape) < 3:
                    # Handle single channel data
                    # Make sure we have a 2D array
                    if len(img.shape) > 2:
                        # Handle unusual shapes
                        img = np.squeeze(img)
                    
                    # For any 1D array, reshape it to 2D
                    if len(img.shape) == 1:
                        side = int(np.sqrt(img.shape[0]))
                        img = img.reshape(side, side)
                    
                    # Ensure proper value range for images
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    
                    # Convert grayscale to RGB if needed
                    return Image.fromarray(img).convert('RGB')
                else:
                    # Standard RGB image processing
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    return Image.fromarray(img)
            else:
                raise TypeError(f"Unsupported input type: {type(img)}")
        
        # Convert all inputs to PIL images
        image_pil = convert_to_pil(images)
        edges_pil = convert_to_pil(edges, is_edge=True)  # Flag this as an edge map
        mask_pil = convert_to_pil(masks)
        
        # Set generator for reproducibility
        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(torch.randint(0, 1000000, (1,)).item())
        
        # Run inpainting
        output_pil = self.pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=image_pil,
            control_image=edges_pil,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            mask_image=mask_pil
        ).images[0]
        
        # Convert PIL image back to tensor
        output_tensor = F.to_tensor(output_pil).float()
        
        # Move to same device as input if needed
        if device is not None:
            output_tensor = output_tensor.to(device)
            
        # Add batch dimension if input had it
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            output_tensor = output_tensor.unsqueeze(0)
            # If it was a batch with multiple images, we'd need to handle that
            # but the current pipeline only processes one image at a time
            
        return output_tensor
