BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 1 \
    --dataset nasbench101 \
    --data_path "$BASE_DIR/data/nasbench101/all_nasbench101.pt" \
    --percent 4236 \
    --batch_size 128 \
    --graph_d_model 192 \
    --d_model 192 \
    --graph_d_ff 768 \
    --graph_n_head 6 \
    --depths 6 1 1 1 \
    --epochs 6000 \
    --model_ema \
    --lr 1e-4 \
    --lambda_diff 0.1 \
    --save_path "output/nasbench201/netaformer_1%/" \
    --embed_type "nerf" \
    --use_extra_token \
    # --aug_data_path "$BASE_DIR/data/nasbench101/OurAug.pt" \
    # --lambda_consistency 0.5 \