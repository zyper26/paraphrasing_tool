from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from prompt import PROMPT_FOR_FALCON7B, PROMPT_FOR_CUSTOM_MODEL
from langchain.llms import GPT4All
import os
from llm import HuggingFaceLLM
from evaluation_metrics import get_bert_score
import time
import gradio as gr
from langchain.llms import VLLM
import torch
import argparse


load_dotenv()


document_dir = os.environ.get('DOCUMENT_DIR')
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")

base_model_type = os.environ.get('BASE_MODEL_TYPE')
base_model_path = os.environ.get('BASE_MODEL_PATH')
base_model_task = os.environ.get('BASE_MODEL_TASK')

custom_model_type = os.environ.get('CUSTOM_MODEL_TYPE')
custom_model_path = os.environ.get('CUSTOM_MODEL_PATH')
custom_model_task = os.environ.get('CUSTOM_MODEL_TASK')

from langchain.document_loaders import (
    PDFMinerLoader,
    TextLoader
)

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".pdf": (PDFMinerLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}


from langchain.docstore.document import Document

def load_single_document(file_path: str) -> Document:
    """
    Load document file (only pdf and text at the moment
    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")

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
    huggingfaceLLM = HuggingFaceLLM(model_name=model_name, task=task, max_length= max_length, \
                                            temperature = temperature, bits_and_bytes_quantize = bits_and_bytes_quantize, \
                                            trust_remote_code = trust_remote_code,\
                                            use_peft_model = use_peft_model)
    huggingfaceLLM.create_pipeline()
    llm = huggingfaceLLM.llm()
    return llm


def get_base_model_output(document_content: str):
    """
        This will return the string output of the generated text on base model i.e. Falcon-7B

    Args:
        document_content (str): Input text
    
    """
    match base_model_type:
        case "HuggingFace":
            s = time.perf_counter()
            
            base_llm = get_hugging_face_models(model_name=base_model_path, task=base_model_task, max_length= 512, \
                                            temperature = 0.9, bits_and_bytes_quantize = False, \
                                            trust_remote_code = True, use_peft_model = False)

            elapsed = time.perf_counter() - s
            print(f"Time For loading the model using HugginFace and creating pipeline: {elapsed:.2f}")

    prompt_falcon = PromptTemplate(input_variables = ['text'], template=PROMPT_FOR_FALCON7B)
    base_chain = LLMChain(llm=base_llm, prompt=prompt_falcon)

    s = time.perf_counter()

    output = base_chain.run(document_content)
    print("Base Model Output: ", output)

    elapsed = time.perf_counter() - s
    print(f"Inference time for base model: {elapsed:.2f}sec")
    return output


def get_custom_model_output(document_content: str):
    """
        This will return the string output of the generated text on custom model.
        From .env file we can make changes on what type of model we want to work with.

    Args:
        document_content (str): Input text
    
    """
    match custom_model_type:
        case "HuggingFace":
            s = time.perf_counter()
            
            custom_llm = get_hugging_face_models(model_name=custom_model_path, task=custom_model_task, max_length= 512, \
                                            temperature = 0, bits_and_bytes_quantize = False, trust_remote_code = True,\
                                            use_peft_model = False)

            elapsed = time.perf_counter() - s
            print(f"Time For loading the model using HugginFace and creating pipeline: {elapsed:.2f}")
        
        case "VLLM":
            s = time.perf_counter()

            custom_llm = VLLM(
                    model=custom_model_path,
                    trust_remote_code=True, 
                    load_in_4bit = True, 
                    max_new_tokens=512,
                    top_k=4,
                    top_p=0.1,
                    temperature=0.01,
            )

            elapsed = time.perf_counter() - s
            print(f"Time For loading the model using HugginFace and creating pipeline: {elapsed:.2f}")

        case "GPT4All":
            s = time.perf_counter()

            cusom_llm = GPT4All(model=custom_model_path, backend='gptj', verbose=False)

            elapsed = time.perf_counter() - s
            print(f"Time For loading the model using HugginFace and creating pipeline: {elapsed:.2f}")        

    prompt = PromptTemplate(input_variables = ['text'], template=PROMPT_FOR_CUSTOM_MODEL)

    custom_chain = LLMChain(llm=custom_llm, prompt=prompt)

    s = time.perf_counter()
    print("this is the document: ", document_content)
    output = custom_chain.run(document_content)
    print("Custom Model Output: " , output)

    elapsed = time.perf_counter() - s
    print(f"Inference time for custom model: {elapsed:.2f}sec")
    return output

def get_output(base_model: bool = False, custom_model: bool = True, text: str = None):
    """
    Function will return the output of both custom_model and base_model based on the arguments.

    Args:
        base_model (bool): If true, base model output will be returned else None will be returned
        custom_model (bool): If true, custom model output will be returned else None will be returned
        text (str): Input text whose paraphrase we want 

    """
    if text == "" or text == None:
        documents = load_single_document(document_dir)
        text = documents.to_json()['kwargs']['page_content']
    if base_model: 
        base_output = get_base_model_output(text)

    if custom_model:
        custom_output = get_custom_model_output(text)

    if base_model and custom_model:
        return base_output, custom_output
    if base_model:
        return base_output, None
    if custom_model:
        return None, custom_output 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", default=False, help="The name of the person.")
    args = parser.parse_args()

    if args.interface:
        demo = gr.Interface(
            fn=get_output,  
            inputs=[
                gr.inputs.Checkbox(label="Base Model (Falcon-7B)"), 
                gr.inputs.Checkbox(label="Custom Model"),
                gr.inputs.Textbox(label="Input Text"), 
            ],
            outputs=[
                gr.outputs.Textbox(label="Base Model output"),
                gr.outputs.Textbox(label="Custom Model output"),
            ],
            live=False,  
            title="Custom Interface",  
            description="Enter text and select boolean values to get results.",
        )
        demo.launch()
    else:
        base_output, custom_output = get_output(base_model = True, custom_model = True)

        if base_output and custom_output:
            print("BERTScore: ", get_bert_score([base_output], [custom_output]))