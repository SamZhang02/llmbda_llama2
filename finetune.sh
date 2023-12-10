python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name saamenerve/mcgill-requisites\
    --load_in_8bit \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2