
def check_llama(model_name_or_path):
    if "vicuna" in model_name_or_path:
        return True
    elif "Llama-2" in model_name_or_path:
        return True
    else:
        return False
    
def check_mistral(model_name_or_path):
    if "Mistral" in model_name_or_path:
        return True
    else:
        return False


def check_llama3(model_name_or_path):
    if "Llama-3" in model_name_or_path:
        return True
    else:
        return False
    
def check_qwen2(model_name_or_path):
    if "Qwen2" in model_name_or_path:
        return True
    else:
        return False