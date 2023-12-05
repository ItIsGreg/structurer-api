import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from structurer_api.utils.prompts import Prompt_List
import ast

router = APIRouter()


# kind of deprecate these responses, need uniform string output
class StructureTextRes(BaseModel):
    substrings: list[str]
    text: str


class StructureTextWithTemplateRes(BaseModel):
    sections_asked_for: Dict[str, list[str]]
    text: str


class StructureTextWithTemplateInferRes(BaseModel):
    sections_asked_for: Dict[str, list[str]]
    sections_inferred: Dict[str, list[str]]
    text: str


class BundleOutlineV2Res(BaseModel):
    outline: Dict[str, list[str]]
    text: str


class StructureTextReq(BaseModel):
    text: str
    api_key: str


class StructureTextWithTemplateReq(BaseModel):
    text: str
    api_key: str
    sections_to_look_for: list[str]


class BundleOutlineV2Req(BaseModel):
    text: str
    api_key: str
    focus_resources: list[str]


class BundleOutLineUnmatchedReq(BaseModel):
    text: str
    api_key: str
    entities: Dict[str, list[Dict[str, str]]]


class BundleOutLineUnmatchedRes(BaseModel):
    entities: Dict[str, list[Dict[str, str]]]
    responseText: str


prompt_list = Prompt_List()


@router.post("/structureText/")
async def structure_text(req: StructureTextReq) -> StructureTextRes:
    """
    Structure a clinical text into meaningful segments.

    Args:
        req (StructureTextReq): text to be structured and OpenAi API key to be used

    Returns:
        StructureTextRes: substrings marking to beginning of each segment
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    segment_text_structured_gpt4 = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured
    )
    result_structured = segment_text_structured_gpt4(req.text)
    result_structured_list = ast.literal_eval(result_structured["text"])

    return StructureTextRes(
        substrings=result_structured_list, text=result_structured["text"]
    )


@router.post("/structureTextWithTemplate/")
async def structure_text_with_template(
    req: StructureTextWithTemplateReq,
) -> StructureTextWithTemplateRes:
    """
    Makes LLM Call to try to find the sections specified in the request in the clinical text.
    Sections not found are supposed to be ignored.
    Text not matched is supposed to be ignored.

    Args:
        req (StructureTextReq): text to be structured, OpenAi API key to be used and sections to look for

    Returns:
        StructureTextRes: substrings marking to beginning of each segment
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    segment_text_structured_gpt4_template = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured_template
    )
    result_structured = segment_text_structured_gpt4_template(
        {"input": req.text, "sections": req.sections_to_look_for}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])

    return StructureTextWithTemplateRes(
        sections_asked_for=result_structured_list, text=result_structured["text"]
    )


@router.post("/structureTextWithTemplateAndInfer/")
async def structure_text_with_template_and_infer(
    req: StructureTextWithTemplateReq,
) -> StructureTextWithTemplateInferRes:
    """
    Makes LLM Call to try to find the sections specified in the request in the clinical text.
    Text that can not be assigned a section from the template should be assigned a section inferred by the LLM

    Args:
        req (StructureTextReq): text to be structured, OpenAi API key to be used and sections to look for

    Returns:
        StructureTextRes: substrings marking to beginning of each segment, divided into sections asked for and sections inferred
    """
    chat_gpt_4 = ChatOpenAI(
        temperature=0, model="gpt-4-1106-preview", openai_api_key=req.api_key
    )
    segment_text_structured_gpt4_template_and_infer = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured_template_and_infer
    )
    result_structured = segment_text_structured_gpt4_template_and_infer(
        {"input": req.text, "sections": req.sections_to_look_for}
    )

    # handle json prefix from gpt-4-turbo
    if result_structured["text"].startswith("```json\n"):
        result_structured["text"] = result_structured["text"][8:]
        # remove end backticks
        result_structured["text"] = result_structured["text"][:-4]

    # result_structured_list = ast.literal_eval(result_structured["text"])
    result_structured_list = json.loads(result_structured["text"])

    return StructureTextWithTemplateInferRes(
        sections_asked_for=result_structured_list["sections_asked_for"],
        sections_inferred=result_structured_list["sections_inferred"],
        # sections_asked_for={"puss": ["puss"]},
        # sections_inferred={"puss": ["puss"]},
        text=result_structured["text"],
    )


@router.post("/structureTextWithTemplateAndInferGerman/")
async def structure_text_with_template_and_infer_german(
    req: StructureTextWithTemplateReq,
) -> StructureTextWithTemplateInferRes:
    """
    Makes LLM Call to try to find the sections specified in the request in the clinical text.
    Text that can not be assigned a section from the template should be assigned a section inferred by the LLM

    Args:
        req (StructureTextReq): text to be structured, OpenAi API key to be used and sections to look for

    Returns:
        StructureTextRes: substrings marking to beginning of each segment, divided into sections asked for and sections inferred
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    segment_text_structured_gpt4_template_and_infer_german = LLMChain(
        llm=chat_gpt_4,
        prompt=prompt_list.segment_text_structured_template_and_infer_german,
    )
    result_structured = segment_text_structured_gpt4_template_and_infer_german(
        {"input": req.text, "sections": req.sections_to_look_for}
    )
    print(result_structured["text"])
    result_structured_list = ast.literal_eval(result_structured["text"])
    print(result_structured_list)
    return StructureTextWithTemplateInferRes(
        sections_asked_for=result_structured_list["sections_asked_for"],
        sections_inferred=result_structured_list["sections_inferred"],
        text=result_structured["text"],
    )


@router.post("/bundleOutlineV2GPT3/")
async def bundleOutlineV2(req: BundleOutlineV2Req) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of fhir-resource types. Makes LLM call to label entities in the text according to resource types.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and resource types to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=req.api_key)
    # bundle_outline_v2_gpt4 = LLMChain(
    #     llm=chat_gpt_4, prompt=prompt_list.bundle_outline_v2
    # )
    bundle_outline_v2_gpt4 = LLMChain(llm=chat, prompt=prompt_list.bundle_outline_v2)
    result_structured = bundle_outline_v2_gpt4(
        {"medical_text": req.text, "focus_resources": req.focus_resources}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])

    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineV2GPT4/")
async def bundleOutlineV2(req: BundleOutlineV2Req) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of fhir-resource types. Makes LLM call to label entities in the text according to resource types.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and resource types to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    # chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=req.api_key)
    bundle_outline_v2_gpt4 = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.bundle_outline_v2
    )
    bundle_outline_v2_gpt4 = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.bundle_outline_v2
    )
    result_structured = bundle_outline_v2_gpt4(
        {"medical_text": req.text, "focus_resources": req.focus_resources}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])

    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineV3GPT3/")
async def bundleOutlineV3(req: BundleOutlineV2Req) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of entites. Makes LLM call to build a resource from the entities.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and entities to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=req.api_key)
    bundle_outline_v3 = LLMChain(llm=chat, prompt=prompt_list.bundle_outline_v3)
    result_structured = bundle_outline_v3(
        {"medical_text": req.text, "entities": req.focus_resources}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])
    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineV3GPT4/")
async def bundleOutlineV3(req: BundleOutlineV2Req) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of entites. Makes LLM call to build a resource from the entities.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and entities to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    bundle_outline_v3 = LLMChain(llm=chat, prompt=prompt_list.bundle_outline_v3)
    result_structured = bundle_outline_v3(
        {"medical_text": req.text, "entities": req.focus_resources}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])
    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineUnmatched")
async def bundleOutlineUnmatched(
    req: BundleOutLineUnmatchedReq, gptVersion: str = "3"
) -> BundleOutLineUnmatchedRes:
    """
    Takes a medical text and a dict of entities with instances of these entities that are paraphrased.
    Makes LLM call to identify exact substring for the paraphrased concepts.
    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and entities to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified concepts
    """
    if gptVersion == "3":
        chat = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=req.api_key
        )
    else:
        chat = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=req.api_key)
    bundle_outline_unmatched = LLMChain(
        llm=chat, prompt=prompt_list.bundle_outline_unmatched
    )
    result_structured = bundle_outline_unmatched(
        {"medical_text": req.text, "entities": req.entities}
    )
    result_structured_list = ast.literal_eval(result_structured["text"])
    return BundleOutLineUnmatchedRes(
        entities=result_structured_list, responseText=result_structured["text"]
    )
