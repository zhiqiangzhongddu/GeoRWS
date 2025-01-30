from code.config import cfg, update_cfg
from code.cv_runner_utils import run_cv_experiments


def main(cfg):
    assert cfg.cv.image_embedding in ['res101_finetuned', 'pretrained_resnet101'], \
        "Available image embeddings."
    assert cfg.cv.class_embedding in ['att', 'wiki2vec', 'cn', 'clip'], \
        "Available class embeddings are att (attributes), wiki2vec, cn (ConceptNet) and clip (CLIP embeddings)."

    print(
        f"Running CV tasks, Method: {cfg.method}. \n"
        f"Dataset: {cfg.dataset}, Embedding: {cfg.cv.image_embedding}, Class: {cfg.cv.class_embedding}"
    )

    run_cv_experiments(cfg=cfg)


if __name__ == "__main__":
    cfg = update_cfg(cfg)

    main(cfg)
