    for known_cls_ratio in 0.25 0.5 0.75
    do
        for shot in 0.05 0.1
        do
                for seed in 15 42 123 987 567 890 456 321 654 789 111 222 333 444 555 666 777 888 999 1000
                do
                    python run.py \
                    --task_name bank \
                    --data_dir data/banking \
                    --shot $shot \
                    --known_ratio $known_cls_ratio\
                    --seed $seed \
                    --rec_drop 0.3 \
                    --rec_num 15 \
                    --batch_size 16 \
                    --val_batch_size 16 \
                    --num_train_epochs 1000 \
                    --learning_rate 2e-5 \
                    --convex \
                    --train_rec \
                    --save_results_path few-shot-nlp-111
        
                done
        done
    done
