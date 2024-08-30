from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
import google.generativeai as genai

from argparse import ArgumentParser
from dotenv import load_dotenv
from tqdm import tqdm
import pathlib
import time
import json
import yaml
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def load_config(config_path: str) -> dict:
    """
    Load the configuration file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_file(file_path: str, join_text: bool = False) -> str|list:
    """
    Reads the file and returns either a list of page or a single string
    """
    loader = PyPDFLoader(file_path)
    document = loader.load()
    if join_text:
        return "\n".join([d.page_content.strip() for d in document])
    return document

def save_file(file_path: str, data: list):
    """
    Save the data to the file
    """
    try:
        with open(file_path, 'w') as f:
            for item in data:
                f.write("%s\n" % item)
        f.close()
    except Exception as e:
        print(e)
        return 
    
def save_json(file_path: str, data: dict):
    """
    Save the data to the file
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    except Exception as e:
        print(e)
        return
    
def split_by_tokens(text: str, chunk_size: int = 4096, chunk_overlap: int = 200) -> list[str]:
    """
    This uses tiktoken library to split the text into tokens
    """
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    return texts

class GenAIModel:
    def __init__(self, **kwargs):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_name = kwargs.get("model_name", "gemini-1.5-flash")
        self.response_mime_type = kwargs.get("response_mime_type", "application/json")
        self.temperature = kwargs.get("temperature", 0.3)
        self.max_output_tokens = kwargs.get("max_output_tokens", 4096)
        self.generate_config = {
            "response_mime_type": self.response_mime_type,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens
        }
        self.model = genai.GenerativeModel(
                            model_name= self.model_name,
                            generation_config=self.generate_config)

    def generate(self, prompt, text):
        response = self.model.generate_content([text, prompt])
        return response.text
        


def main():
    parser = ArgumentParser(
                prog='Question-Answer Generation from Documents',
                description='This script generates question-answer pairs from a document.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size for text splitting in tokens')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Token Chunk overlap for text splitting')
    parser.add_argument('--model_name', type=str, default="gemini-1.5-flash", help='Model name to use for generation')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for generation')
    parser.add_argument('--max_output_tokens', type=int, default=4096, help='Maximum output tokens for generation')
    parser.add_argument('--response_mime_type', type=str, default="application/json", help='Response mime type for generation')
    parser.add_argument('--join_text', type=bool, default=True, help='Join the text from the document')

    args = parser.parse_args()
    config = load_config("config.yaml")
    prompt = config.get("prompt")    


    ##### Read the file and split the text into tokens #####
    document = read_file(file_path = args.input_file, join_text=args.join_text)
    texts = split_by_tokens(text = document, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    file_name = pathlib.Path(args.input_file.split(".")[0] + "_chunks.json")
    print(f"Saving the raw text to {file_name}")
    save_json(file_path=file_name, data=texts)
    
    ##### Generate the question-answer pairs #####
    model = GenAIModel(**vars(args))
    n_requests = 0
    for text in tqdm(texts):
        if n_requests % 15 == 0:
            print("Sleeping for 1 minutes to avoid rate limit")
            time.sleep(60)
        try:
            response = model.generate(prompt, text)
            with open(args.output_file, "a") as f:
                f.write(response)
                f.write("\n")
        except Exception as e:
            print(e)
            continue
        n_requests += 1
    print("Done")



if __name__ == "__main__":
    main()