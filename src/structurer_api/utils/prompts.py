from langchain import PromptTemplate


class Template_List:
    def __init__(self):
        self.bundle_outline_v2 = """
        You are an assistant to a researcher creating FHIR-Data. You will be provided with a medical text. The task for you and the researcher is to create a bundle of FHIR-resources,
        representing the content of the text. As a first step to creating these resources, your task is to create an outline of this bundle. The output should have the following form:

        {{
            "ResourceType1": [
                "Concept1",
                "Concept2",
            ],
            "ResourceType2": [
                "Concept3",
            ]
        }}
        
        Make sure to keep the output pure without further explanation.
        Make sure only to use resource types from the FHIR R4 specification.
        Try to capture as many concepts as possible in resource form.

        If possible try to extract resources specified in the focus resources. If there is no concept corresponding to a requested focus resource, Leave it out.
        If there are concepts not corresponding to a requested focus resource, leave them out.
        %FOCUS RESOURCES
        {focus_resources}

        %MEDICAL TEXT:
        {medical_text}
    """

        self.bundle_outline_unmatched = """
        You will be provided with a medical text. Furthermore, you will be provided with a input data structure, looking like this:

        {{
            "Entity1": [
                {{"ParaphrasedConcept1": ""}},
                {{"ParaphrasedConcept2": ""}}
            ],
            "Entity2": [
                {{"ParaphrasedConcept3": ""}},
            ]
        }}

        Your task is to identify the paraphrased concepts in the text and find the corresponding substring. The exact substring should be found.

        The output should have the following form:

        {{
            "Entity1": [
                {{"ParaphrasedConcept1": "ExactSubstring1"}},
                {{"ParaphrasedConcept2": "ExactSubstring2"}}
            ],
            "Entity2": [
                {{"ParaphrasedConcept3": "ExactSubstring3"}},
            ]
        }}

        Make sure to keep the output pure without further explanation.

        %ENTITIES:
        {entities}

        %TEXT:
        {medical_text}
"""
        self.bundle_outline_unmatched_with_attributes = """
        You will be provided with a medical text. Furthermore, you will be provided with a input data structure, looking like this:

        {{
            "Entity1": [
                {{"ParaphrasedConcept1": ""
                    "attributes": {{
                        "attribute1": "value1",
                        "attribute2": "value2"
                    }}
                    }}
                }},
                {{"ParaphrasedConcept2": ""
                    "attributes": {{
                        "attribute3": "value3",
                        "attribute4": "value4"
                    }}
                    }}
                }}
            ],
            "Entity2": [
                {{"ParaphrasedConcept3": ""
                    "attributes": {{
                        "attribute5": "value5",
                        "attribute6": "value6"
                    }}
                    }}
                }},
            ]
        }}

        Your task is to identify the paraphrased concepts in the text and find the corresponding substring. The exact substring should be found.
        The attributes should normally be ok, but if you find a mistake, please correct it.

        The output should have the following form:

        {{
            "Entity1": [
                {{"ParaphrasedConcept1": "ExactSubstring1"
                    "attributes": {{
                        "attribute1": "value1",
                        "attribute2": "value2"
                    }}
                    }}
                }},
                {{"ParaphrasedConcept2": "ExactSubstring2"
                    "attributes": {{
                        "attribute3": "value3",
                        "attribute4": "value4"
                    }}
                    }}
                }}
                }},
            ],
            "Entity2": [
                {{"ParaphrasedConcept3": "ExactSubstring3"
                    "attributes": {{
                        "attribute5": "value5",
                        "attribute6": "value6"
                    }}
                    }}
                }},
                }},
            ]
        }}

        Make sure to keep the output pure without further explanation.

        %ENTITIES:
        {entities}

        %TEXT:
        {medical_text}
"""
        self.bundle_outline_v3 = """
        You will be provided with a medical text. Furthermore, you will be provided with a list of entities.
        Your task is to identify instances of the entities in the text and return the corresponding substring from the text. The exact substring should be returned, without paraphrasing.
        The entities might be representing FHIR-resource types.
        The output should have the following form:

        {{
            "Entity1": [
                "Substring1",
                "Substring2",
            ],
            "Entity2": [
                "Substring3",
            ]
        }}

        Make sure to keep the output pure without further explanation.

        %ENTITIES:
        {entities}

        %Entity Descriptions:
        Condition: A condition/ diagnosis for a patient. E.g. "Diabetes", "Hypertension", "Pacemaker", "Pregnancy", "Coronary Bypass".
        Observation: Observations are results of observations, measurements, and assessments performed on a patient. This could include vital signs, laboratory results, imaging findings, device measurements, clinical scores or other types of clinical observations. 
        Medication: A Medication is a definable product that is intended to be administered to a patient to realize beneficial or therapeutic effects. E.g. "Aspirin", "Metformin", "Insulin", "Lisinopril", "Atorvastatin", ...
        Procedure: A procedure is a procedure or intervention that has been performed on a patient. Examples include surgical procedures, diagnostic procedures, endoscopic procedures, biopsies, counseling, physiotherapy.
        %MEDICAL TEXT:
        {medical_text}
"""
        self.build_resource_v3 = """
        You are an assistant to a researcher creating FHIR-Data. You will be provided with
            1) a FHIR-Resource-Type
            2) a medical term
            3) the text passage the term is extracted from
        Your task is to create a FHIR-R4-Resource of specified resource type representing the medical term.
        The output should be in json format without further explanation.
        
        %FHIR-RESOURCE-TYPE:
        {resource_type}

        %MEDICAL TERM:
        {medical_term}

        %TEXT PASSAGE:
        {context}
    """
        self.correct_json_error = """
        You are an assistant to a researcher creating FHIR-Data. You will be provided with
            1) a FHIR-Resource-Type
            2) a medical term
            3) the text passage the term is extracted from
        Your task is to create a FHIR-R4-Resource of specified resource type representing the medical term.
        The output should be in json format without further explanation.
        
        %FHIR-RESOURCE-TYPE:
        {resource_type}

        %MEDICAL TERM:
        {medical_term}

        %TEXT PASSAGE:
        {context}

        The first attempt to create a resource failed. Please make sure to only output a valid FHIR resource and nothing else.
    """
        self.correct_validation_error = """
        You are an assistant to a researcher creating FHIR-Data. You will be provided with
            1) a FHIR-Resource-Type
            2) a medical term
            3) the text passage the term is extracted from
        Your task is to create a FHIR-R4-Resource of specified resource type representing the medical term.
        The output should be in json format without further explanation.
        
        %FHIR-RESOURCE-TYPE:
        {resource_type}

        %MEDICAL TERM:
        {medical_term}

        %TEXT PASSAGE:
        {context}

        Your first attempt produced the following resource:
        {resource}

        Validation of this resource failed with following error:
        {error}

        Please modify the resource to comply with the error message.
    """
        self.segment_text_structured = """
        Please divide the text provided into segments. For each segment provide a substring at the beginning of the segment, at least 20 characters long.
        Do not leave out any part of the text; e.g. each part of the text should be a part of a segment. There should be no overlap between the segments. Subsegments are allowed.
        Keep the order of the segments as they appear in the text.

        %TEXT:
        {input}

        The output should look like this:
        [substring_1, substring_2, ...]
        """
        self.segment_text_structured_template = """
    Please identify the sections specified in the text provided.
    If, in your opinion a section is present provide a substring that marks the beginning of the section, about 20 characters long and substring that marks the end of the section, also about 20 characters long.
    %SECTIONS:
    {sections}

    %TEXT:
    {input}

    The output should look like this:
    {{
        "section_1": [substring_1, substring_2],
        "section_2": [substring_1, substring_2],
    }}
    Make sure to keep the output pure without further explanation.

    """
        self.segment_text_structured_template_and_infer = """
    Please identify the sections specified further down in the text provided further down.
    If, in your opinion a section is present provide a substring that marks the beginning of the section, about 20 characters long and substring that marks the end of the section, also about 20 characters long.
    If a section is not present, leave it out.
    If parts of the text remain that do not fit in any of the sections provided, provide section headings for these parts and provide start and end substring for these parts.
    Multiple newline characters should be in the end of a section.

    %SECTIONS:
    {sections}

    %TEXT:
    {input}

    The output should look like this:
    {{
        "sections_asked_for": {{
            "section_1": [substring_1, substring_2],
            "section_2": [substring_1, substring_2]
        }},
        "sections_inferred": {{
            "section_3": [substring_1, substring_2],
            "section_4": [substring_1, substring_2]
        }}
    }}
    Make sure to keep the output pure without further explanation.

    """
        self.segment_text_structured_template_and_infer_german = """
    Bitte identifizieren Sie die weiter unten angegebenen Abschnitte im ebenfalls weiter unten angegebenen Text.
    Wenn Sie der Meinung sind, dass ein Abschnitt vorhanden ist, geben Sie einen Substring an, der den Beginn des Abschnitts markiert, etwa 20 Zeichen lang, und einen Substring, der das Ende des Abschnitts markiert, ebenfalls etwa 20 Zeichen lang.
    Wenn ein Abschnitt nicht vorhanden ist, lassen Sie ihn aus.
    Wenn Teile des Textes übrig bleiben, die in keinen der angegebenen Abschnitte passen, geben Sie Überschriften für diese Teile an und geben Sie Anfangs- und End-Substrings für diese Teile an.

    %SECTIONS:
    {sections}

    %TEXT:
    {input}

    Das Ausgabeformat sollte wie folgt aussehen:
    {{
        "sections_asked_for": {{
            "section_1": [substring_1, substring_2],
            "section_2": [substring_1, substring_2]
        }},
        "sections_inferred": {{
            "section_3": [substring_1, substring_2],
            "section_4": [substring_1, substring_2]
        }}
    }}
    Stellen Sie sicher, dass die Ausgabe rein ohne weitere Erklärung bleibt.
"""
        self.bundle_outline_with_attributes = """
        You are an assistant to a researcher creating FHIR-Data. You will be provided with a medical text. The task for you and the researcher is to create a bundle of FHIR-resources,
        representing the content of the text. As a first step to creating these resources, your task is to create an outline of this bundle.
        The outline is a dictionary of dictionaries of dictionaries. The first level of the dictionary is the resource type. The resource types to look for are specified further down.
        The second level of the dictionary is the concept. The concepts to look for are specified further down. In the output, the concepts should not be paraphrased, but the exact substring should be returned.
        The third level of the dictionary comprises of attributes of the concept. The attributes to look for are specified further down.
        The output should have the following form:

        {{
            "ResourceType1": {{
                "Substring_Concept1": {{
                    "Attribute1": "Value1",
                    "Attribute2": "Value2"
                }},
                "Substring_Concept2": {{
                    "Attribute3": "Value3",
                    "Attribute4": "Value4"
                }},
            }},
            "ResourceType2": {{
                "Substring_Concept3": {{
                    "Attribute5": "Value5",
                    "Attribute6": "Value6"
                }},
            }}
        }}
        
        Make sure to keep the output pure without further explanation.
        Try to capture as many concepts as possible in resource form.

        OUTPUT Example:

        {{
            "Condition": {{
                "Diabetes": {{
                    "ClinicalStatus": "Active",
                    "VerificationStatus": "Confirmed"
                }},
            }},
        }}

        If possible try to extract resources specified in the focus resources and try to fill out the attributes to look for as well. If there is no concept corresponding to a requested focus resource, Leave it out.
        If there are concepts not corresponding to a requested focus resource, leave them out.
        %FOCUS RESOURCES
        {focus_resources}

        %MEDICAL TEXT:
        {medical_text}
        """


class Prompt_List:
    def __init__(self):
        template_list = Template_List()
        self.bundle_outline_v2 = PromptTemplate(
            input_variables=["medical_text", "focus_resources"],
            template=template_list.bundle_outline_v2,
        )
        self.bundle_outline_v3 = PromptTemplate(
            input_variables=["medical_text", "focus_resources"],
            template=template_list.bundle_outline_v3,
        )
        self.bundle_outline_with_attributes = PromptTemplate(
            input_variables=["medical_text", "focus_resources"],
            template=template_list.bundle_outline_with_attributes,
        )
        self.build_resource_v3 = PromptTemplate(
            input_variables=["medical_term", "context", "resource_type"],
            template=template_list.build_resource_v3,
        )
        self.correct_json_error = PromptTemplate(
            input_variables=["medical_term", "context", "resource_type"],
            template=template_list.correct_json_error,
        )
        self.correct_validation_error = PromptTemplate(
            input_variables=[
                "medical_term",
                "context",
                "resource_type",
                "resource",
                "error",
            ],
            template=template_list.correct_validation_error,
        )
        self.segment_text_structured = PromptTemplate(
            input_variables=["input"],
            template=template_list.segment_text_structured,
        )
        self.segment_text_structured_template = PromptTemplate(
            input_variables=["input", "sections"],
            template=template_list.segment_text_structured_template,
        )
        self.segment_text_structured_template_and_infer = PromptTemplate(
            input_variables=["input", "sections"],
            template=template_list.segment_text_structured_template_and_infer,
        )
        self.segment_text_structured_template_and_infer_german = PromptTemplate(
            input_variables=["input", "sections"],
            template=template_list.segment_text_structured_template_and_infer_german,
        )
        self.bundle_outline_unmatched = PromptTemplate(
            input_variables=["medical_text", "entities"],
            template=template_list.bundle_outline_unmatched,
        )
        self.bundle_outline_unmatched_with_attributes = PromptTemplate(
            input_variables=["medical_text", "entities"],
            template=template_list.bundle_outline_unmatched_with_attributes,
        )
