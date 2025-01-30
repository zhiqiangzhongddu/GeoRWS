import torch
import clip


def prepare_vocab(opt, matcontent, zst_mode = False):
    vocab = []
    if zst_mode:
        dataset = opt.zstfrom
    else:
        dataset = opt.dataset
    for cls_name in matcontent['allclasses_names']:
        if dataset == 'CUB':
            vocab.append(cls_name[0][0][4:])
        elif dataset == 'SUN':
            vocab.append(cls_name[0][0])
        elif dataset == 'AWA2':
            vocab.append(cls_name[0][0].replace('+', '_'))
        else:
            raise NotImplementedError

    return vocab


def prep_imagenet_vocab(imagenet_vocab):
    pruned_vocab = []
    for label in imagenet_vocab:
        first_label = label.split(",")[0]
        pruned_vocab.append(first_label)
    return pruned_vocab


def get_clip_embeddings(vocab):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN101", device=device)

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

    return embeddings
