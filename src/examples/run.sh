# python main.py --dataset_name dynhate_sample --model_name fastchat-t5-ft-bn-dynhate-5ep-v1-all-gpt --binary_only --all_gpt
# python main.py --dataset_name sbic_sample --model_name fastchat-t5-ft-bn-sbic-5ep-v1-all-gpt --binary_only --all_gpt
# python main.py --dataset_name dynhate_sample --model_name fastchat-t5-ft-full-dynhate-5ep-v1-all-gpt-base --all_gpt
# python main.py --dataset_name sbic_sample --model_name fastchat-t5-ft-full-sbic-5ep-v1-all-gpt-base --all_gpt
python main.py --dataset_name sbic_sample --model_name fastchat-t5-ft-full-sbic-5ep-v1-all-gpt --use_dtot --all_gpt
python main.py --dataset_name dynhate_sample --model_name fastchat-t5-ft-full-dynhate-5ep-v1-all-gpt --use_dtot --all_gpt



