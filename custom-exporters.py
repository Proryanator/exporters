from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from exporters.coreml import export
from exporters.coreml.models import CustomLlamaCoreMLConfig

model_ckpt = "meta-llama/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_ckpt, torchscript=True)
preprocessor = AutoTokenizer.from_pretrained(model_ckpt)

# do we need to use past here? Since llama2 is a decoder only model?
# coreml_config = CustomLlamaCoreMLConfig(base_model.config, task="text-generation", use_past=True)

# use_past throws an error: For mlprogram, inputs with infinite upper_bound is not allowed. Please set upper_bound to a positive value in "RangeDim()" for the "inputs" param in ct.convert().
coreml_config = CustomLlamaCoreMLConfig(base_model.config, task="text-generation")
mlmodel = export(preprocessor, base_model, coreml_config, quantize="float16")

# modify the output to not include the shape?
# mlmodel.output_description

mlmodel.save('outputs/llama2-chat-hf.mlpackage')
