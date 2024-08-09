from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from exporters.coreml import export
from exporters.coreml.models import CustomLlamaCoreMLConfig

model_ckpt = "meta-llama/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_ckpt, torchscript=True)
preprocessor = AutoTokenizer.from_pretrained(model_ckpt)

coreml_config = CustomLlamaCoreMLConfig(base_model.config, task="text-generation")
mlmodel = export(preprocessor, base_model, coreml_config)
mlmodel.save('outputs/llama2-chat-hf.mlpackage')
