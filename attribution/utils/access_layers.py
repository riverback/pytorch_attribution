import torch

def replace_layer_recursive(model: torch.nn.Module, old_layer: torch.nn.Module, new_layer: torch.nn.Module):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


def replace_all_layer_type_recursive(model: torch.nn.Module, old_layer_type: torch.nn.Module, new_layer: torch.nn.Module):
    '''new_layer is a instance of the new layer type, not the type itself.'''
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer
        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


def find_layer_types_recursive(model: torch.nn.Module, layer_types: torch.nn.Module):
    def predicate(layer):
        return type(layer) in layer_types
    return find_layer_predicate_recursive(model, predicate)


def find_layer_predicate_recursive(model: torch.nn.Module, predicate):
    result = []
    for name, layer in model._modules.items():
        if predicate(layer):
            result.append(layer)
        result.extend(find_layer_predicate_recursive(layer, predicate))
    return result
