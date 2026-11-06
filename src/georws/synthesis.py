import torch


def synthesize_weights(model, a_rows, w_ref, batch_size, unseen_scale, device):
    if unseen_scale not in ('none', 'seen_norm', 'icis'):
        raise ValueError(f'Unknown unseen scale: {unseen_scale}')
    model.eval()
    weights, alphas = [], []
    with torch.no_grad():
        for a in a_rows.split(batch_size):
            logits, _, _ = model(a.to(device), w_ref)
            alpha = logits.float().softmax(dim=1)
            w_new = alpha @ w_ref
            if unseen_scale == 'seen_norm':
                w_new = w_new * (w_ref.norm(dim=1).mean() / w_new.norm(dim=1, keepdim=True).clamp_min(1e-12))
            weights.append(w_new.cpu())
            alphas.append(alpha.cpu())
    return torch.cat(weights), torch.cat(alphas)
