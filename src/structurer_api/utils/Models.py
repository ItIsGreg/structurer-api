from pydantic import BaseModel
from typing import Dict


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


class BundleOutlineWithAttributesRes(BaseModel):
    outline: Dict[str, Dict[str, Dict[str, str]]]
    # Dict[resource_type, list[Dict[concept_substring, Dict[attribute, attribute_value]]]]
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


class FocusResourceWithAttributes(BaseModel):
    resource_type: str
    attributes: list[str]


class BundleOutlineWithAttributesReq(BaseModel):
    text: str
    api_key: str
    focus_resources: list[FocusResourceWithAttributes]


class BundleOutLineUnmatchedReq(BaseModel):
    text: str
    api_key: str
    entities: Dict[str, list[Dict[str, str]]]


class BundleOutLineUnmatchedRes(BaseModel):
    entities: Dict[str, list[Dict[str, str]]]
    responseText: str