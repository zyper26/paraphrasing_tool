from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
from transformers import BitsAndBytesConfig
from peft import PeftConfig, PeftModel

def get_bnb_config():
    return BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )

class HuggingFaceLLM():
    def __init__(self, model_name: str, task: str = None, max_length : int = 200, \
                    temperature: int = 0.01, bits_and_bytes_quantize: bool = False, trust_remote_code: bool = True, \
                    use_peft_model: bool = False):
        self.model_name = model_name
        self.task = task
        self.max_length = max_length
        self.temperature = temperature
        self.bits_and_bytes_quantize = bits_and_bytes_quantize
        self.trust_remote_code = trust_remote_code
        self.use_peft_model = use_peft_model

    def create_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.task == "text2text-generation":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.task == "text-generation":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.bits_and_bytes_quantize:
            self.model.quantization_config = get_bnb_config()
        if self.use_peft_model:
            self.model = PeftModel.from_pretrained(self.model, self.model_name)
       
        self.pipeline = pipeline(
            self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            device_map="auto",
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def llm(self):
        return HuggingFacePipeline(pipeline = self.pipeline, model_kwargs = { 'temperature': self.temperature })
