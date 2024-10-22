# Helper functions and classes
import os
from tqdm import tqdm
import json
import logging
from ..utils.helper import ProjectPath
from typing import Literal
from pydantic import BaseModel, Field
from IPython.display import display, Markdown
import json

# File processing
from .file import File

# LLM Interation Layer
from .model_invoke_iterative import ModelInvoker


# Pydantic Models
class ContractMetadata(BaseModel):
    language: Literal["FR", "EN"] = Field(
        description= (
            "The language in which the document is written in. "
            "Example if the language detected is French return 'FR' and only that. "
            "If the language is English return 'EN' and only that."
        )
    )


class CraftAIExtractor:
    def __init__(
        self,
        pydantic_prompt_en,
        pydantic_prompt_fr,
        gcp_project_name,
        corp_sec_approval_y_n,
    ):
        self.language_config = {"EN": pydantic_prompt_en, "FR": pydantic_prompt_fr}
        self.gcp_project_name = gcp_project_name
        self.corp_sec_approval_y_n = corp_sec_approval_y_n
        self.current_file_name = None
        logging.info(f"Initializing Vertex AI with project: {gcp_project_name}")

    def get_json(self, path_file):
        self.current_file_name = os.path.basename(path_file)
        file = File(path_file)
        language = self.detect_language(file)
        if language not in ["EN", "FR"]:
            logging.info("No valid language found!")
            logging.info("Continuing extraction with English language.")
            language = "EN"
        try:
            return self.extract(file, language)
        except Exception as e:
            logging.exception(f"File not processed: {e}")

    def get_bulk_json(self, input_folder, path_save_folder):
        # Define supported file extensions
        supported_extensions = {
            ".msg",
            ".txt",
            ".pdf",
            ".docx",
            ".doc",
            ".xls",
            ".xlsx",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".tif",
        }

        files = os.listdir(input_folder)
        for file in tqdm(files, desc="Processing files"):
            file_extension = os.path.splitext(file)[1]
            if file_extension not in supported_extensions:
                continue

            file_path = ProjectPath(input_folder) + f"/{file}"
            json_file_name = os.path.splitext(file)[0] + ".json"
            json_response = self.get_json(file_path)
            if json_response:
                self.save_json(json_response, path_save_folder, json_file_name)

    def detect_language(self, file):
        prompt = (
			"You are a data steward representative for a Telco Company. "
            "You are tasked with detecting the language and relevancy from a contract. "
			"Ignore signatures and email fields. "
			"Return only 'FR' or 'EN' for the language and only that."
		)
        model_invoker = ModelInvoker(
            pydantic_object=ContractMetadata,
            prompt=prompt,
            file=file,
            gcp_project_name=self.gcp_project_name,
            corp_sec_approval_y_n=self.corp_sec_approval_y_n,
        )
        language_metadata = model_invoker.invoke_model()
        return language_metadata.language

    def extract(self, file, language):
        prompt, pydantic_object = self.language_config[language]
        if prompt == None or pydantic_object == None:
            if language == "FR":
                prompt, pydantic_object = self.language_config["EN"]
            else:
                prompt, pydantic_object = self.language_config["FR"]
        if prompt == None or pydantic_object == None:
            print("No valid pydantic prompt provided !")
        model_invoker = ModelInvoker(
            pydantic_object=pydantic_object,
            prompt=prompt,
            file=file,
            gcp_project_name=self.gcp_project_name,
            corp_sec_approval_y_n=self.corp_sec_approval_y_n,
        )
        response_extract = model_invoker.invoke_model()
        return response_extract.dict()

    def save_json(self, json_response, path_save_folder, json_file_name=None):
        # add logic to make sure that we dont load into a json already containing data
        if json_file_name == None:
            json_file_name = os.path.splitext(self.current_file_name)[0] + ".json"
        output_file = path_save_folder + json_file_name
        try:
            with open(output_file, "w") as outfile:
                json.dump(json_response, outfile, indent=2)
        except Exception as e:
            print(f"Error saving document: {e}")

    def display_json(self, json_response):
        if not isinstance(json_response, dict):
            with open(json_response, "r") as file:
                json_response = json.load(file)
        display(Markdown("```json\n" + json.dumps(json_response, indent=2) + "\n```"))


def get_answer(prompt, path_file, gcp_project_name, corp_sec_approval_y_n):
    if corp_sec_approval_y_n != "Y":
        raise ValueError(
            'corp_sec_approval_y_n is missing. Please indicate "Y" if you received corporate security approval to send your files to Gemini.'
        )
    file = File(path_file)
    try:
        logging.info(f"Initializing Vertex AI with project: {gcp_project_name}")
        model_invoker = ModelInvoker(
            pydantic_object=None,
            prompt=prompt,
            file=file,
            gcp_project_name=gcp_project_name,
            corp_sec_approval_y_n=corp_sec_approval_y_n,
        )
        return model_invoker.invoke_model()
    except Exception as e:
        logging.exception(f"File not processed: {e}")
