import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
topk = 20
here = ''