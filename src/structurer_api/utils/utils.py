def handle_json_prefix(result_structured):
    if result_structured["text"].startswith("```json\n"):
        result_structured["text"] = result_structured["text"][8:]
        # remove end backticks
        result_structured["text"] = result_structured["text"][:-4]
    return result_structured
