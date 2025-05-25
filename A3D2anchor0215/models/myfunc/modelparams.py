
def num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Count the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1000000, trainable_params/1000000
