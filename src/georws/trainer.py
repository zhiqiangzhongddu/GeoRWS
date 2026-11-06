import torch
import torch.nn.functional as F

from src.georws.losses import georws_losses


def train_georws(model, a_seen, w_seen, cfg, device, log_prefix, eval_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=(cfg.train.beta1, 0.999))
    history, updates, patience, last_losses = [], 0, 0, {}
    model.train()
    for epoch in range(1, cfg.train.num_epoch + 1):
        totals = {}
        for m_cpu in torch.randperm(a_seen.size(0)).split(cfg.train.batch_size):
            m = m_cpu.to(device)
            a = a_seen[m_cpu].to(device)
            logits, _, recon = model(a, w_seen)
            logits_tilde = None
            if cfg.georws.use_cons:
                logits_tilde, _, _ = model(F.dropout(a, p=cfg.georws.aug_dropout, training=True), w_seen)
            losses = georws_losses(logits, m, w_seen, a, recon, cfg, logits_tilde)
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            updates += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item() * len(m_cpu)
        last_losses = {name: value / a_seen.size(0) for name, value in totals.items()}
        history.append(last_losses['total'])
        if eval_fn is not None and cfg.zsl.eval_every > 0 and epoch % cfg.zsl.eval_every == 0:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            try:
                model.eval()
                with torch.no_grad():
                    eval_fn(epoch)
            finally:
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                model.train()
        if cfg.georws.early_stopping and epoch > 20:
            slope = -(sum(history[-10:]) / 10 - sum(history[-20:-10]) / 10) / 10
            patience = patience + 1 if slope < cfg.georws.early_stopping_threshold else 0
        stop = cfg.georws.early_stopping and patience >= 5
        if epoch % cfg.georws.log_every == 0 or epoch == cfg.train.num_epoch or stop:
            print(f'{log_prefix} epoch={epoch} updates={updates} ' +
                  ' '.join(f'{k}={v:.6g}' for k, v in last_losses.items()))
        if stop:
            break
    return dict(epochs=epoch, updates=updates, last_losses=last_losses)
