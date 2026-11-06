import torch


def perturb_affinity(a, kind, rate, sigma, generator):
    if kind == 'none':
        return a
    if kind == 'embedding_dropout':
        mask = torch.rand(a.shape, generator=generator, device=a.device) >= rate
        return a * mask / (1 - rate)
    if kind == 'gaussian_noise':
        noise = torch.randn(a.shape, generator=generator, device=a.device, dtype=a.dtype)
        return a + sigma * a.norm(dim=-1, keepdim=True) * noise / noise.norm(dim=-1, keepdim=True)
    if kind == 'pair_swap':
        flat = a.reshape(-1, a.size(-1))
        count = flat.size(0)
        n = round(rate * count)
        indices = torch.randperm(count, generator=generator, device=a.device)[:n]
        sources = torch.randint(count, (n,), generator=generator, device=a.device)
        collision = sources == indices
        while collision.any():
            sources[collision] = torch.randint(count, (int(collision.sum()),), generator=generator, device=a.device)
            collision = sources == indices
        result = flat.clone()
        result[indices] = flat[sources]
        return result.reshape_as(a)
    raise ValueError(f'Unknown perturbation: {kind}')
