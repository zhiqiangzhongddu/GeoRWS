import torch
import torch.nn.functional as F


def gather_references(logits, masked, w):
    batch, count = logits.shape
    assert count >= 2
    ids = torch.arange(count, device=logits.device).expand(batch, -1)
    keep_idx = ids[ids != masked[:, None]].reshape(batch, count - 1)
    log_alpha = F.log_softmax(logits.float().gather(1, keep_idx), dim=1)
    alpha = log_alpha.exp()
    w_ref = w[keep_idx]
    w_hat = (alpha[..., None] * w_ref).sum(dim=1)
    return keep_idx, log_alpha, alpha, w_ref, w_hat


def scatter_alpha(alpha, keep_idx, count):
    return alpha.new_zeros(alpha.size(0), count).scatter(1, keep_idx, alpha)


def georws_losses(logits, masked, w, a, recon, cfg, logits_tilde=None):
    keep, log_alpha, alpha, w_ref, w_hat = gather_references(logits, masked, w)
    zero = logits.new_zeros((), dtype=torch.float32)
    align = ((w_hat - w[masked]) ** 2).sum(dim=1).mean()
    recon_loss = zero
    if recon is not None:
        idx = keep[..., None].expand(-1, -1, a.size(-1))
        per_pair = ((a.float().gather(1, idx) - recon.float().gather(1, idx)) ** 2).sum(dim=-1)
        if cfg.georws.recon_reduction == 'sum':
            recon_loss = per_pair.sum(dim=1).mean()
        elif cfg.georws.recon_reduction == 'mean':
            recon_loss = per_pair.mean(dim=1).mean()
        else:
            raise ValueError(f'Unknown reconstruction reduction: {cfg.georws.recon_reduction}')
    geo = zero
    if cfg.georws.use_geo:
        assert cfg.georws.tau_w > 0
        with torch.no_grad():
            cos = F.cosine_similarity(w[masked, None, :-1], w_ref[..., :-1], dim=-1) / cfg.georws.tau_w
            log_p = F.log_softmax(cos.float(), dim=1)
            p = log_p.exp()
        geo = (p * (log_p - log_alpha)).sum(dim=1).mean()
    cons = zero
    if cfg.georws.use_cons:
        log_t = F.log_softmax(logits_tilde.float().gather(1, keep), dim=1)
        alpha_t = log_t.exp()
        w_tilde = (alpha_t[..., None] * w_ref).sum(dim=1)
        cons = (((w_hat - w_tilde) ** 2).sum(dim=1) +
                (alpha * (log_alpha - log_t) + alpha_t * (log_t - log_alpha)).sum(dim=1)).mean()
    total = align + cfg.georws.lam * recon_loss + cfg.georws.beta * geo + cfg.georws.gamma * cons
    return dict(total=total, align=align, recon=recon_loss, geo=geo, cons=cons)
