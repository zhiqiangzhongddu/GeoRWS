import re
import numpy as np
import torch


def raw_class_names(mat):
    names = []
    for entry in mat['allclasses_names'].reshape(-1):
        while isinstance(entry, np.ndarray):
            entry = entry.flat[0]
        names.append(str(entry))
    return names


def class_names(dataset, mat):
    names = raw_class_names(mat)
    if dataset == 'CUB':
        return [re.sub(r'^\d+\.', '', name).replace('_', ' ') for name in names]
    if dataset == 'AWA2':
        return [name.replace('+', ' ') for name in names]
    if dataset == 'SUN':
        return [name.replace('_', ' ') for name in names]
    raise NotImplementedError(dataset)


def get_clip_embeddings(vocab):
    import clip

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('RN101', device=device)
    input_text = []
    prompt = 'a photo of a'
    for word in vocab:
        word = word.replace('_', ' ')
        word = word.lower()
        if word[0] in ['a', 'e', 'i', 'o', 'A', 'E', 'I', 'O']:
            input_text.append(prompt + 'n ' + word)
        else:
            input_text.append(prompt + ' ' + word)
    text = clip.tokenize(input_text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    embeddings = text_features / torch.norm(text_features.float(), dim=-1).unsqueeze(-1)
    return embeddings.float().cpu()
