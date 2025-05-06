import copy
import torch

def fed_avg(state_dicts):
    """
    Federated averaging of a list of state_dicts.
    """
    global_dict = copy.deepcopy(state_dicts[0])
    for key in global_dict.keys():
        global_dict[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return global_dict


def fed_prox(state_dicts, global_dict, mu=0.1):
    """
    FedProx aggregation: includes proximal term adjustment.
    """
    agg = copy.deepcopy(global_dict)
    num_clients = len(state_dicts)
    for key in agg.keys():
        delta = torch.stack([sd[key].float() - global_dict[key].float() for sd in state_dicts], dim=0).sum(dim=0) / num_clients
        agg[key] = (global_dict[key].float() + delta) / (1 + mu)
    return agg
