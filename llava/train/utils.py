
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