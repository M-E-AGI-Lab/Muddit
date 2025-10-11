# bash it in root path
PYTHON_PATH='./' accelerate launch --multi_gpu --gpu_ids '0,1,2,3,4,5,6,7' --main_process_port 25000 --num_processes 8 train/train_unified.py \
        --output_dir "/path/to/output/dir" \
        --train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --text_loss_weight 0 \
        --max_grad_norm 10 \
        --pretrained_model_name_or_path "MeissonFlow/Meissonic" \
        --pretrained_transformer_path "MeissonFlow/Meissonic" \
        --text_encoder_architecture 'open_clip' \
        --instance_dataset 'ImageCaptionLargeDataset' \
        --instance_data_dir  '/path/to/data/' \
        --resolution 512 \
        --mixed_precision bf16 \
        --lr_scheduler constant \
        --use_8bit_adam \
        --dataloader_num_workers 4 \
        --validation_prompts \
            'a boy' \
            'A serene mountain landscape with towering snow-capped peaks, a crystal-clear blue lake reflecting the mountains, dense pine forests, and a vibrant orange sunrise illuminating the sky.' \
            'A playful golden retriever puppy with a shiny coat, bounding through a meadow filled with colorful wildflowers, under a bright, clear blue sky.' \
            'A bustling city street at night, illuminated by vibrant neon signs in various colors, with busy pedestrians, street vendors, and a light rain creating reflective puddles on the pavement.' \
            'A majestic, medieval castle perched on a rugged cliffside, overlooking a vast, calm ocean at sunset, with the sky painted in hues of pink, orange, and purple.' \
            'An elegant ballerina in a white tutu, dancing gracefully on a grand stage with ornate, gold-trimmed curtains, under a spotlight that casts a soft glow.' \
            'A cozy, rustic log cabin nestled in a snow-covered forest, with smoke rising from the stone chimney, warm lights glowing from the windows, and a path of footprints leading to the front door.'\
            'A Cute Cat' \
            'A Snow Mountain'\
        --max_train_steps 100000 \
        --checkpointing_steps 1000 \
        --validation_steps 100 \
        --report_to 'wandb' \
        --logging_steps 10
