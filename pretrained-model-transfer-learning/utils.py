
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