from langchain_google_vertexai import ChatVertexAI
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain.globals import set_verbose
set_verbose(True)
from .file import File

import logging
import vertexai

class ModelInvoker:
    def __init__(self, pydantic_object: BaseModel, prompt: str, file: File, model_name: str = "gemini-1.5-pro-001", gcp_project_name: str = None, corp_sec_approval_y_n: str = None, temperature: float = 0.2, max_tokens: int = 8192):
        # logging.info("Initializing ModelInvoker")
        self.pydantic_object = pydantic_object
        self.prompt = prompt
        self.file = file

        if corp_sec_approval_y_n != "Y":
            raise ValueError('CORP_SEC_APPROVAL_Y_N is missing. Please indicate "Y" if you received corporate security approval to send your files to Gemini.')
        
        # logging.info(f"Initializing Vertex AI with project: {gcp_project_name}")
        vertexai.init(project=gcp_project_name, location="us-central1")
        self.model = ChatVertexAI(model_name=model_name,
                                  max_tokens=max_tokens,
                                  temperature=temperature)
        
        if self.pydantic_object != None:
            self.parser = OutputFixingParser.from_llm(
                parser=PydanticOutputParser(pydantic_object=self.pydantic_object), llm=self.model, max_retries=10
            )
        else: self.parser = None
    

    def invoke_model(self):
        # logging.info("Invoking the language model with the message")
        try:
            mime_type = self.file.file_metadata.mime_type
            if not mime_type:
                logging.warning("mime_type is missing or null. Proceeding without mime_type.")
                mime_type = None
        except KeyError as e:
            logging.error(f"KeyError: {e}. Proceeding without mime_type.")
            mime_type = None


        if self.parser is None:
            text_content = self.prompt
        else :
            text_content = self.parser.get_format_instructions()
            text_content = f"{self.prompt}\n{text_content}"        
        
        message_content = [
            {
                "type": "text",
                "text": text_content,
            }
        ]

        if mime_type:
            # logging.info(f"Sending document of mime_type: {mime_type}")
            message_content.append(
                {
                    "type": "media",
                    "mime_type": mime_type,
                    "data": self.file.base64
                }
            )
        if self.file.children_file_metadata :
            for child in getattr(self.file, "children_file_metadata", []):
                mime_type=getattr(child, "mime_type", None)
                with open(child.base64_file_path, "r") as f:
                    file_content = f.read()
                message_content.append(
                    {"type": "media", "mime_type": mime_type, "data": file_content}
                )


        else:
            # logging.info("No mime_type provided, sending text content only.")
            pass
        message = HumanMessage(content=message_content)

        if self.parser is None:
            return self.model.invoke([message]).content
        else:
            chain = self.model | self.parser
            return chain.invoke([message])