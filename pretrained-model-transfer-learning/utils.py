import os
import json

def load_classnames_json(file_path):
    imagenet_classes = None
    if os.path.exists(file_path):
        try:
            # Open and load the JSON file
            with open(file_path, 'r') as f:
                imagenet_classes = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"JSON file not found")
        
    return imagenet_classes


def show_classes_from_weight_meta(weights_obj) -> tuple[list|None, int|None]:
    """
    Retrieve class numbers and class names from model weight meta.
    """
    num_classes = None
    class_names = None

    if weights_obj and hasattr(weights_obj, 'meta') and "categories" in weights_obj.meta:
        class_names = weights_obj.meta["categories"]
        num_classes = len(class_names)
    else:
        print("Weights categories is not detected")
    
    return class_names, num_classes
