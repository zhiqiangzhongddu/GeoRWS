# Geometry-Regularized Relational Weight Synthesis

## Instruction to execution

Before everything, you need to create ``openai_env.py`` file to indicate your ``OPENAI_API_KEY="xxx"``. 

1. Generate class descriptions.
   1. Generated descriptions are located in (``input/descriptions``).
2. Encode class description into Embeddings 
   1. Generated embeddings are located in (``input/embeddings``).
3. Run Zero-shot methods
   1. Run baselines
   2. Run Our method


## Example Execution Code

### Generate class descriptions
Example: 
```
./description_generator.sh # generate class descriptions for all datasets
python -m code.descrition_generator 
```

### Encode class description into Embeddings
Example: 
```
python -m text_embedding.py dataset SUN text_encoder BERT
python -m text_embedding.py dataset SUN text_encoder T5
python -m text_embedding.py dataset SUN text_encoder CLIP
```

### Run Zero-shot methods
Example: 
```
# ICIS
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method icis cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
# ConSE
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method conse cv.image_embedding res101_finetuned cv.class_embedding att cv.conse_benchmark True
# COSTA
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method costa cv.image_embedding res101_finetuned cv.class_embedding att cv.costa_benchmark True
# Sub. Reg.
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method subreg cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.subspace_proj True
# wDAE
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wdae cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.daegnn True
# WAvg
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wavg cv.image_embedding res101_finetuned cv.vgse_baseline wavg cv.class_embedding att norm_scale_heuristic True
# SMO
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method smo cv.image_embedding res101_finetuned cv.vgse_baseline smo cv.class_embedding att cv.vgse_alpha 0. norm_scale_heuristic True
```

### Testing
```
# Single
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method asci cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method comc cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method icis cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method conse cv.image_embedding res101_finetuned cv.class_embedding att cv.conse_benchmark True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method costa cv.image_embedding res101_finetuned cv.class_embedding att cv.costa_benchmark True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method subreg cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.subspace_proj True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wdae cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.daegnn True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wavg cv.image_embedding res101_finetuned cv.vgse_baseline wavg cv.class_embedding att norm_scale_heuristic True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method smo cv.image_embedding res101_finetuned cv.vgse_baseline smo cv.class_embedding att cv.vgse_alpha 0. norm_scale_heuristic True
# Group
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method asci num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method comc num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method icis num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cos_sim_loss True include_unseen True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.calc_entropy True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method conse  num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cv.conse_benchmark True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method costa num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cv.costa_benchmark True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method subreg num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.subspace_proj True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wdae  num_runs 5 cv.image_embedding res101_finetuned cv.class_embedding att cv.single_autoencoder_baseline True num_layers 2 beta1 0.9 lr 0.00001 batch_size 16 embed_dim 2048 strict_eval True early_stopping_slope True cv.daegnn True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method wavg num_runs 5 cv.image_embedding res101_finetuned cv.vgse_baseline wavg cv.class_embedding att norm_scale_heuristic True
CUDA_VISIBLE_DEVICES=1 python -m code.cv_runner dataset SUN method smo num_runs 5 cv.image_embedding res101_finetuned cv.vgse_baseline smo cv.class_embedding att cv.vgse_alpha 0. norm_scale_heuristic True
```
