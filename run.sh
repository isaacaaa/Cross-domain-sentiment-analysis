python3 final_cross_domain_uda_mtl.py \
  --do_train \
  --do_eval \
  --num_unlabelled 10000 \
  --per_gpu_suptrain_batch_size 4 \
  --per_gpu_unsuptrain_batch_size 16 \
  --per_gpu_simtrain_batch_size 4 \
  --learning_rate 2e-5 \
  --weights 0.1 \
  --sim_weights 0.1 \
  --num_train_epochs 15 \
  --max_seq_length 256 \
  --output_dir proposed/ 
 
 
