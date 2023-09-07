
# Paraphrase Generation

A tool to experiment different LLMs capability for paraphrase generation.

We can choose whether we want to compare the two models prediction or we need only one to be predicted.

As base, we have falcon-7B model.
For custom model, we have 3 different pipelines setup: 


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```
python main.py --interface False
```

| Argument            | Description                                      |
|---------------------|--------------------------------------------------|
| `-i` or `--interface`    | Gradio Interface True or False                       |


## Evaluation Metrics

[BERTScore](https://arxiv.org/pdf/1904.09675.pdf)

# Configuration

For base model:

* HuggingFace: [Models](https://huggingface.co/models)


For custom models we have 3 pipelines in the code:

* HuggingFace: [Models](https://huggingface.co/models)
* [VLLM](https://vllm.ai/)
* GPT4ALL: [Models](https://gpt4all.io/index.html) Look for Model Explorer section in the page to know which all models can be take for this


Below function arguments needs to be changed in `main.py` based on the requirements and optimizations: 

```
def get_hugging_face_models(model_name: str, task: str, max_length: int= 512, \
                            temperature: float = 0.1, bits_and_bytes_quantize: bool = False, \
                            trust_remote_code: bool = True, use_peft_model: bool = False):
    """
    Function for the creating the LLM pipeline using HuggingFace

    Args:
        model_name (str): Model path (huggingface or local model weights)
        task (str): Model task based on model which needs to be used (text-generation or text2text-generation or any other type)\ 
                    For now only text-generation and text2text-generation is supported
        max_length (int): Max context length
        temperature (float): temperature to control the hallucinations
        bit_and_bytes_quantize (bool): For bits and bytes quantization
        trust_remote_code (bool): True to load model from huggingface
        use_peft_model (bool): True in case if you have peft finetuned model 

    """
```

# WARNING
* Don't use Falcon Model locally until you have high computation power machine 


### Future Works:
* Try more prompts
* FineTune models on [dataset](http://nlpprogress.com/english/paraphrase-generation.html) with prompt tuning







