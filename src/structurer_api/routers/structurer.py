from html import entities
import json
from fastapi import APIRouter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from structurer_api.utils.Models import (
    BundleOutLineUnmatchedReq,
    BundleOutLineUnmatchedRes,
    BundleOutlineUnmatchedWithAttributesReq,
    BundleOutlineUnmatchedWithAttributesRes,
    BundleOutlineV2Req,
    BundleOutlineV2Res,
    BundleOutlineWithAttributesReq,
    BundleOutlineWithAttributesRes,
    StructureTextReq,
    StructureTextRes,
    StructureTextWithTemplateInferRes,
    StructureTextWithTemplateReq,
    StructureTextWithTemplateRes,
)
from structurer_api.utils.prompts import Prompt_List
from structurer_api.utils.utils import handle_json_prefix

router = APIRouter()

prompt_list = Prompt_List()


@router.post("/structureText/")
async def structure_text(
    req: StructureTextReq, gptModel: str = "gpt-3.5-turbo"
) -> StructureTextRes:
    """
    Structure a clinical text into meaningful segments.

    Args:
        req (StructureTextReq): text to be structured and OpenAi API key to be used

    Returns:
        StructureTextRes: substrings marking to beginning of each segment
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    segment_text_structured_gpt4 = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured
    )
    result_structured = segment_text_structured_gpt4(req.text)
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])

    return StructureTextRes(
        substrings=result_structured_list, text=result_structured["text"]
    )


@router.post("/structureTextWithTemplate/")
async def structure_text_with_template(
    req: StructureTextWithTemplateReq, gptModel: str = "gpt-3.5-turbo"
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
    chat_gpt_4 = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    segment_text_structured_gpt4_template = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured_template
    )
    result_structured = segment_text_structured_gpt4_template(
        {"input": req.text, "sections": req.sections_to_look_for}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])

    return StructureTextWithTemplateRes(
        sections_asked_for=result_structured_list, text=result_structured["text"]
    )


@router.post("/structureTextWithTemplateAndInfer/")
async def structure_text_with_template_and_infer(
    req: StructureTextWithTemplateReq, gptModel: str = "gpt-3.5-turbo"
) -> StructureTextWithTemplateInferRes:
    """
    Makes LLM Call to try to find the sections specified in the request in the clinical text.
    Text that can not be assigned a section from the template should be assigned a section inferred by the LLM

    Args:
        req (StructureTextReq): text to be structured, OpenAi API key to be used and sections to look for

    Returns:
        StructureTextRes: substrings marking to beginning of each segment, divided into sections asked for and sections inferred
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    segment_text_structured_gpt4_template_and_infer = LLMChain(
        llm=chat_gpt_4, prompt=prompt_list.segment_text_structured_template_and_infer
    )
    result_structured = segment_text_structured_gpt4_template_and_infer(
        {"input": req.text, "sections": req.sections_to_look_for}
    )

    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)

    result_structured_list = json.loads(result_structured["text"])

    return StructureTextWithTemplateInferRes(
        sections_asked_for=result_structured_list["sections_asked_for"],
        sections_inferred=result_structured_list["sections_inferred"],
        text=result_structured["text"],
    )


@router.post("/structureTextWithTemplateAndInferGerman/")
async def structure_text_with_template_and_infer_german(
    req: StructureTextWithTemplateReq, gptModel: str = "gpt-3.5-turbo"
) -> StructureTextWithTemplateInferRes:
    """
    Makes LLM Call to try to find the sections specified in the request in the clinical text.
    Text that can not be assigned a section from the template should be assigned a section inferred by the LLM

    Args:
        req (StructureTextReq): text to be structured, OpenAi API key to be used and sections to look for

    Returns:
        StructureTextRes: substrings marking to beginning of each segment, divided into sections asked for and sections inferred
    """
    chat_gpt_4 = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    segment_text_structured_gpt4_template_and_infer_german = LLMChain(
        llm=chat_gpt_4,
        prompt=prompt_list.segment_text_structured_template_and_infer_german,
    )
    result_structured = segment_text_structured_gpt4_template_and_infer_german(
        {"input": req.text, "sections": req.sections_to_look_for}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])

    return StructureTextWithTemplateInferRes(
        sections_asked_for=result_structured_list["sections_asked_for"],
        sections_inferred=result_structured_list["sections_inferred"],
        text=result_structured["text"],
    )


@router.post("/bundleOutlineV2/")
async def bundleOutlineV2(
    req: BundleOutlineV2Req, gptModel: str = "gpt-3.5-turbo"
) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of fhir-resource types. Makes LLM call to label entities in the text according to resource types.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and resource types to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    bundle_outline_v2_gpt4 = LLMChain(llm=chat, prompt=prompt_list.bundle_outline_v2)
    result_structured = bundle_outline_v2_gpt4(
        {"medical_text": req.text, "focus_resources": req.focus_resources}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])

    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineV3/")
async def bundleOutlineV3(
    req: BundleOutlineV2Req, gptModel: str = "gpt-3.5-turbo"
) -> BundleOutlineV2Res:
    """
    Takes a medical text and a list of entites. Makes LLM call to build a resource from the entities.

    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and entities to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified entities
    """
    chat = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    bundle_outline_v3 = LLMChain(llm=chat, prompt=prompt_list.bundle_outline_v3)
    result_structured = bundle_outline_v3(
        {"medical_text": req.text, "entities": req.focus_resources}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])
    return BundleOutlineV2Res(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineWithAttributes/")
async def bundleOutlineWithAttributes(
    req: BundleOutlineWithAttributesReq, gptModel: str = "gpt-3.5-turbo"
) -> BundleOutlineWithAttributesRes:
    """
    Takes a medical text and a list of fhir-resource types together with attributes.
    Makes LLM call to label entities in the text according to resource types and extract attributes.
    Args:
        req (BundleOutlineWithAttributesReq): text to be structured, OpenAi API key to be used and resource types to look for, with attributes to extract
    Returns:
        BundleOutlineWithAttributesRes: Dict with of resource types with list of dict of substrings representing the identified entities and dict of extracted attributes
    """
    chat = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    bundle_outline_with_attributes = LLMChain(
        llm=chat, prompt=prompt_list.bundle_outline_with_attributes
    )
    result_structured = bundle_outline_with_attributes(
        {"medical_text": req.text, "focus_resources": req.focus_resources}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])
    return BundleOutlineWithAttributesRes(
        outline=result_structured_list, text=result_structured["text"]
    )


@router.post("/bundleOutlineUnmatched/")
async def bundleOutlineUnmatched(
    req: BundleOutLineUnmatchedReq, gptModel: str = "gpt-3.5-turbo"
) -> BundleOutLineUnmatchedRes:
    """
    Takes a medical text and a dict of entities with instances of these entities that are paraphrased.
    Makes LLM call to identify exact substring for the paraphrased concepts.
    Args:
        req (BundleOutlineV2Req): text to be structured, OpenAi API key to be used and entities to look for
    Returns:
        BundleOutlineV2Res: Dict with of resource types with list of substrings representing the identified concepts
    """
    chat = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    bundle_outline_unmatched = LLMChain(
        llm=chat, prompt=prompt_list.bundle_outline_unmatched
    )
    result_structured = bundle_outline_unmatched(
        {"medical_text": req.text, "entities": req.entities}
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])
    return BundleOutLineUnmatchedRes(
        entities=result_structured_list, responseText=result_structured["text"]
    )


@router.post("/bundleOutlineUnmatchedWithAttributes/")
async def bundleOutlineUnmatchedWithAttributes(
    req: BundleOutlineUnmatchedWithAttributesReq, gptModel: str = "gpt-3.5-turbo"
) -> BundleOutlineUnmatchedWithAttributesRes:
    """
    Takes a medical text and a dict of entities with instances of these entities that are paraphrased.
    Makes LLM call to identify exact substring for the paraphrased concepts.
    Args:
        req (BundleOutlineUnmatchedWithAttributesReq): text to be structured, OpenAi API key to be used and paraphrased entities to find the exact substring for, with attributes already extracted
    Returns:
        BundleOutlineUnmatchedWithAttributesRes: Dict of resource types, with list of dict of substrings representing the identified concepts and dict of extracted attributes
    """
    chat = ChatOpenAI(temperature=0, model=gptModel, openai_api_key=req.api_key)
    bundle_outline_unmatched_with_attributes = LLMChain(
        llm=chat, prompt=prompt_list.bundle_outline_unmatched_with_attributes
    )
    result_structured = bundle_outline_unmatched_with_attributes(
        {
            "medical_text": req.text,
            "entities": req.entities,
        }
    )
    # handle json prefix from gpt-4-turbo
    result_structured = handle_json_prefix(result_structured)
    result_structured_list = json.loads(result_structured["text"])
    return BundleOutlineUnmatchedWithAttributesRes(
        entities=result_structured_list, responseText=result_structured["text"]
    )
