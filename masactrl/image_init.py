import cv2
import numpy as np
import torch
import einops
import PIL
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import inspect

from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models import MultiAdapter,T2IAdapter
from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple



logger = logging.get_logger(__name__)

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)

class sketch_canny:
    
    def __init__(self, adapter: Union[T2IAdapter, MultiAdapter, List[T2IAdapter]]):  
        self.adapter = adapter
        self.register_modules(adapter=adapter)
        
        pass
    
    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)
            
            
    def register_to_config(self, **kwargs):
        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)
    
    def preprocess_adapter_image(self, image, height, width):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
            image = [
                i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
            ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            if image[0].ndim == 3:
                image = torch.stack(image, dim=0)
            elif image[0].ndim == 4:
                image = torch.cat(image, dim=0)
            else:
                raise ValueError(
                    f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
                )
        return image
    
    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[-2]

            # round down to nearest multiple of `self.adapter.downscale_factor`
            height = (height // self.adapter.downscale_factor) * self.adapter.downscale_factor

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[-1]

            # round down to nearest multiple of `self.adapter.downscale_factor`
            width = (width // self.adapter.downscale_factor) * self.adapter.downscale_factor

        return height, width
    
    def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        
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
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def canny(self, images , batch_size , channels , height, width):
        apply_canny = CannyDetector()
        imgs4=[]
        # img = np.transpose(img, (2,0,1))
        for img in images:
            
            img = cv2.resize(img, (width//8, height//8))
            
            detected_map = apply_canny(img, 100, 200)
            detected_map = HWC3(detected_map)
            
            
            img4=cv2.cvtColor(detected_map,cv2.COLOR_BGR2BGRA)
            
            h,w=detected_map.shape[:2]
            for y in range(h):
                for x in range(w):
                    b,g,r=detected_map[y,x]
                    a=255
                    rgba=(r,g,b,a)
                    img4[y,x]=rgba
                    
            img4 = 255 - img4
                    
            imgs4.append(img4)
            

        
        img = np.array(imgs4)  
        #print(img.shape)
        
        control = torch.from_numpy(img.copy()).float() / 255.0
        # control = torch.stack([control for _ in range(batch_size)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
            
        return control
    
    def sketch2canny(self, images , batch_size , channels , height, width):
        apply_canny = CannyDetector()
        imgs4=[]
        count=0
        # img = np.transpose(img, (2,0,1))
        for img in images:
            
            img = cv2.resize(img, (width, height))
            
            detected_map = apply_canny(img, 100, 200)
            detected_map = HWC3(detected_map)
                    
            # detected_map = 255 - detected_map   
            
            imgs4.append(detected_map)  
            
            #cv2.imwrite(f'./in/sketch2canny_{count}.png', detected_map)
        
        
        img = np.array(imgs4)  
        #print(img.shape)
        
        # control = torch.from_numpy(img.copy()).float() / 255.0
        # control = torch.stack([control for _ in range(batch_size)], dim=0)
        # control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    
        
            
        return img
    
    
    def i2l(self, images , batch_size , channels , height, width, num_inference_steps, timesteps=None):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        height, width = self._default_height_width(height, width, images)
        
        adapter_input = []
        for one_image in images:
            one_image = self.preprocess_adapter_image(one_image, height, width)
            one_image = cv2.resize(one_image, (width//8, height//8))
            # one_image = one_image.to(device=device, dtype=self.adapter.dtype)
            one_image = np.transpose(one_image, (2,0,1))
            adapter_input.append(one_image)
        adapter_input=torch.Tensor(adapter_input)
        adapter_input = adapter_input.to(device=device, dtype=self.adapter.dtype)
        
        adapter_conditioning_scale=1.0
        # adapter_state = self.adapter(adapter_input, adapter_conditioning_scale)
        adapter_state = self.adapter(adapter_input)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v
            
        img_square = [state.clone() for state in adapter_state]
        return img_square 