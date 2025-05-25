
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import copy



# class LoRALayer(nn.Module):
#     def __init__(self, original_layer, lora_r, lora_alpha):
#         super(LoRALayer, self).__init__()
#         self.lora_r = lora_r
#         self.lora_alpha = lora_alpha
#         self.original_layer = original_layer
#         # Ensure compatibility with the original layer's dimensions
#         self.lora_A = nn.Parameter(torch.randn(self.lora_r, original_layer.in_features))
#         self.lora_B = nn.Parameter(torch.randn(original_layer.out_features, self.lora_r))
#         # Initialize parameters
#         nn.init.normal_(self.lora_A, std=0.01)
#         nn.init.normal_(self.lora_B, std=0.01)

#     def forward(self, x):
#         # Compute LoRA's output
#         lora_output = x @ self.lora_A.t() @ self.lora_B.t()
#         # Compute the original layer's output
#         original_output = self.original_layer(x)
#         # Combine outputs with the scaling factor alpha
#         return original_output + lora_output * self.lora_alpha


# class ModelWithLoRA(nn.Module):
#     def __init__(self, model, lora_r, lora_alpha, is_bert=False):
#         super(ModelWithLoRA, self).__init__()
#         self.is_bert = is_bert
#         self.lora_r = lora_r
#         self.lora_alpha = lora_alpha
#         self.model = model
#         # Freeze all parameters in the original model
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#         # for param in self.model.classifier.parameters():
#         #     param.requires_grad = True

#         # Apply LoRA to the model
#         self._apply_lora(self.model)

#     def _apply_lora(self, module):
#         """
#         Recursively applies LoRA adaptation to all nn.Linear layers within the model.
#         """
#         for name, child in module.named_children():
#             if isinstance(child, nn.Linear):
#                 # Replace the nn.Linear layer with a LoRALayer
#                 setattr(module, name, LoRALayer(child, self.lora_r, self.lora_alpha))
#             else:
#                 # If the child is not an nn.Linear layer, apply LoRA recursively to its children
#                 self._apply_lora(child)

#     def forward(self, *x, **xs):
#         # Forward pass through the adapted model
#         if self.is_bert:
#             output = self.model(**xs)
#         else: 
#             #print('MMMMMMMMMMMMMMMMMM:', x)
#             #output = self.model(pixel_values=x[0])
#             output = self.model(x[0])
#             if hasattr(self.model, 'logits'):
#                 self.logits = self.model.logits
#         return output




import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original_layer, lora_r, lora_alpha):
        super(LoRALayer, self).__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.original_layer = original_layer
        # Ensure compatibility with the original layer's dimensions
        self.lora_A = nn.Parameter(torch.randn(self.lora_r, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.randn(original_layer.out_features, self.lora_r))
        # Initialize parameters
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.01)

    def forward(self, x):
        # Compute LoRA's output
        lora_output = x @ self.lora_A.t() @ self.lora_B.t()
        # Compute the original layer's output
        original_output = self.original_layer(x)
        # Combine outputs with the scaling factor alpha
        return original_output + lora_output * self.lora_alpha


class ModelWithLoRA(nn.Module):
    def __init__(self, model, lora_r, lora_alpha, is_bert=False):
        super(ModelWithLoRA, self).__init__()
        self.is_bert = is_bert
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.model = model
        # Freeze all parameters in the original model
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA to the model
        self._apply_lora(self.model)

    def _apply_lora(self, module):
        """
        Recursively applies LoRA adaptation to all nn.Linear layers within the model.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace the nn.Linear layer with a LoRALayer, retaining the original name
                setattr(module, name, LoRALayer(child, self.lora_r, self.lora_alpha))
            else:
                # If the child is not an nn.Linear layer, apply LoRA recursively to its children
                self._apply_lora(child)

    def forward(self, *x, **xs):
        # Forward pass through the adapted model
        if self.is_bert:
            output = self.model(**xs)
        else:
            output = self.model(x[0])
            if hasattr(self.model, 'logits'):
                self.logits = self.model.logits
        return output

    def __getattr__(self, name):
        # Delegate attribute access to the original model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)




######################################################################################
def apply_lora_adjustments(original_model, lora_model):
    """
    Apply trained LoRA adjustments from lora_model to the original_model.
    This function assumes that corresponding layers in both models are matched by order.
    """
    original_linear_layers = [module for module in original_model.modules() if isinstance(module, nn.Linear)]
    lora_layers = [module for module in lora_model.modules() if isinstance(module, LoRALayer)]

    # Ensure there's a matching number of layers to adjust
    assert len(original_linear_layers) == len(lora_layers), "Mismatch in the number of Linear and LoRALayer modules"

    with torch.no_grad():  # Ensure no gradients are computed during this process
        for original_layer, lora_layer in zip(original_linear_layers, lora_layers):
            # Compute the LoRA adjustment
            lora_adjustment = lora_layer.lora_A.t() @ lora_layer.lora_B.t() * lora_layer.lora_alpha
            lora_adjustment = lora_adjustment.t() 
            
            # Apply the adjustment to the original layer's weight
            original_layer.weight += lora_adjustment
    
    return original_model

#################################################################################

