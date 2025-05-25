
import torch 
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from .mylora import ModelWithLoRA, apply_lora_adjustments


# class Config():
#     def __init__(self):
#         self.num_labels = 7  # 7 emotion categories
#         self.percent = 0.2
#         self.batch_size = 64*5
#         self.path = "/data3/sunjun/work/code/PEFT2/data/RAFDB/"

#         self.is_lora = 1
#         self.lora_alpha = 1
#         self.lora_r = 8
#         self.is_bert = 0

#         self.learning_rate = 1e-4 #2e-4
#         self.epochs = 2
#         self.amp = 0
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config = Config()
###################








def lora_apvit(model, r):
            
    lora_alpha = 1
    lora_r = r #32
    is_bert = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelWithLoRA(model, lora_r, lora_alpha, is_bert).to(device)

    return model
##############################################

def lora_apvit(model, r):
            
    lora_alpha = 1
    lora_r = r #32
    is_bert = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelWithLoRA(model, lora_r, lora_alpha, is_bert).to(device)

    return model
##############################################

def lora_apvit2(model, r):

    lora_alpha = 1 
    lora_r = r #32
    is_bert = 0
    layer_s = 6
    layer_e = 8
  
    tm = [
        f"vit.blocks.{i}.attn.qkv"
        for i in range(layer_s, layer_e)
    ] + [
        f"vit.blocks.{i}.attn.proj"
        for i in range(layer_s, layer_e)
    ] + [
        f"vit.blocks.{i}.mlp.fc1"
        for i in range(layer_s, layer_e)
    ] + [
        f"vit.blocks.{i}.mlp.fc2"
        for i in range(layer_s, layer_e)
    ]

    lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            #target_modules="all-linear",
            target_modules=tm,
            lora_dropout=0.1,
            bias="none",
            #task_type="CLASSIFICATION",
            #quantization='int8'
        )
    model = get_peft_model(model, lora_config)
    return model
#############################################

def lora_vit(model, r):

    lora_alpha = 1 
    lora_r = r #32
    is_bert = 0
    layer_s = 10
    layer_e = 12

    lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            #target_modules=["query", "key", "value", "dense", "classifier"],
            #target_modules="all-linear",
            # target_modules=[
            #         # "encoder.layer.0.attention.self.query", "encoder.layer.0.attention.self.key", "encoder.layer.0.attention.self.value", "encoder.layer.0.attention.output.dense",
            #         # "encoder.layer.1.attention.self.query", "encoder.layer.1.attention.self.key", "encoder.layer.1.attention.self.value", "encoder.layer.1.attention.output.dense",
            #         # "encoder.layer.2.attention.self.query", "encoder.layer.2.attention.self.key", "encoder.layer.2.attention.self.value", "encoder.layer.2.attention.output.dense",
            #         # "encoder.layer.3.attention.self.query", "encoder.layer.3.attention.self.key", "encoder.layer.3.attention.self.value", "encoder.layer.3.attention.output.dense",
            #         "encoder.layer.4.attention.self.query", "encoder.layer.4.attention.self.key", "encoder.layer.4.attention.self.value", "encoder.layer.4.attention.output.dense",
            #         "encoder.layer.5.attention.self.query", "encoder.layer.5.attention.self.key", "encoder.layer.5.attention.self.value", "encoder.layer.5.attention.output.dense",
            #         "encoder.layer.6.attention.self.query", "encoder.layer.6.attention.self.key", "encoder.layer.6.attention.self.value", "encoder.layer.6.attention.output.dense",
            #         "encoder.layer.7.attention.self.query", "encoder.layer.7.attention.self.key", "encoder.layer.7.attention.self.value", "encoder.layer.7.attention.output.dense",
            #         "encoder.layer.8.attention.self.query", "encoder.layer.8.attention.self.key", "encoder.layer.8.attention.self.value", "encoder.layer.8.attention.output.dense",
            #         "encoder.layer.9.attention.self.query", "encoder.layer.9.attention.self.key", "encoder.layer.9.attention.self.value", "encoder.layer.9.attention.output.dense",
            #         "encoder.layer.10.attention.self.query", "encoder.layer.10.attention.self.key", "encoder.layer.10.attention.self.value", "encoder.layer.10.attention.output.dense",
            #         "encoder.layer.11.attention.self.query", "encoder.layer.11.attention.self.key", "encoder.layer.11.attention.self.value", "encoder.layer.11.attention.output.dense",
            #         # "encoder.layer.0.attention.output.dense", "encoder.layer.0.intermediate.dense", "encoder.layer.0.output.dense",
            #         # "encoder.layer.1.attention.output.dense", "encoder.layer.1.intermediate.dense", "encoder.layer.1.output.dense",
            #         # "encoder.layer.2.attention.output.dense", "encoder.layer.2.intermediate.dense", "encoder.layer.2.output.dense",
            #         # "encoder.layer.3.attention.output.dense", "encoder.layer.3.intermediate.dense", "encoder.layer.3.output.dense",
            #         "encoder.layer.4.attention.output.dense", "encoder.layer.4.intermediate.dense", "encoder.layer.4.output.dense",
            #         "encoder.layer.5.attention.output.dense", "encoder.layer.5.intermediate.dense", "encoder.layer.5.output.dense",
            #         "encoder.layer.6.attention.output.dense", "encoder.layer.6.intermediate.dense", "encoder.layer.6.output.dense",
            #         "encoder.layer.7.attention.output.dense", "encoder.layer.7.intermediate.dense", "encoder.layer.7.output.dense",
            #         "encoder.layer.8.attention.output.dense", "encoder.layer.8.intermediate.dense", "encoder.layer.8.output.dense",
            #         "encoder.layer.9.attention.output.dense", "encoder.layer.9.intermediate.dense", "encoder.layer.9.output.dense",
            #         "encoder.layer.10.attention.output.dense", "encoder.layer.10.intermediate.dense", "encoder.layer.10.output.dense",
            #         "encoder.layer.11.attention.output.dense", "encoder.layer.11.intermediate.dense", "encoder.layer.11.output.dense",
            #         "classifier"],

            target_modules=[f"encoder.layer.{i}.attention.self.{proj}" for i in range(layer_s, layer_e) for proj in ["query", "key", "value", "output.dense"]] 
                         + [f"encoder.layer.{i}.attention.output.dense" for i in range(layer_s, layer_e)] 
                         + [f"encoder.layer.{i}.intermediate.dense" for i in range(layer_s, layer_e)] 
                         + [f"encoder.layer.{i}.output.dense" for i in range(layer_s, layer_e)] 
                         + ["classifier"],
            lora_dropout=0.1,
            bias="none",
            #task_type="CLASSIFICATION",
            #quantization='int8'
        )
    model = get_peft_model(model, lora_config)
    return model
#############################################################


def lora_bert(model, r):
    lora_alpha = 1 * 5 # * 2 #* 5
    lora_r = r #32
    layer_s = 10  #4
    layer_e = 12

    lora_config = LoraConfig(
    #task_type=TaskType.SEQ_CLS, 
    r=lora_r, 
    lora_alpha=lora_alpha, 
    lora_dropout=0.1,
    #target_modules="all-linear",
    target_modules=[f"encoder.layer.{i}.attention.self.{proj}" for i in range(layer_s, layer_e) for proj in ["query", "key", "value", "output.dense"]] 
                 + [f"encoder.layer.{i}.intermediate.dense" for i in range(layer_s, layer_e)] 
                 + [f"encoder.layer.{i}.output.dense" for i in range(layer_s, layer_e)]
                 + ["pooler.dense"],
    )

    model = get_peft_model(model, lora_config)

    return model

##################################################

# def lora_wav2vec(model):
#     lora_alpha = 1 * 2
#     lora_r = 16

#     lora_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             #target_modules=["query", "key", "value", "dense", "classifier"],
#             target_modules="all-linear",
#             lora_dropout=0.1,
#             bias="none",
#             #task_type="CLASSIFICATION",
#             #quantization='int8'
#         )
#     model = get_peft_model(model, lora_config)
#     return model

def lora_wav2vec(model, r):
    lora_alpha = 1 * 2
    lora_r = r #32
    layer_s = 10 #4 
    layer_e = 12

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        #target_modules="all-linear",
        target_modules=[f"encoder.layers.{i}.attention.{proj}" for i in range(layer_s, layer_e) for proj in ["k_proj", "v_proj", "q_proj", "out_proj"]] 
                    + [f"encoder.layers.{i}.feed_forward.{dense}" for i in range(layer_s, layer_e) for dense in ["intermediate_dense", "output_dense"]],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model



def lora_wavlm(model, r):
    lora_alpha = 1 * 2
    lora_r = r #32
    layer_s = 10 #4 
    layer_e = 12

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        #target_modules="all-linear",
        target_modules=[f"encoder.layers.{i}.attention.{proj}" for i in range(layer_s, layer_e) for proj in ["k_proj", "v_proj", "q_proj", "out_proj"]] 
                    + [f"encoder.layers.{i}.feed_forward.{dense}" for i in range(layer_s, layer_e) for dense in ["intermediate_dense", "output_dense"]],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model