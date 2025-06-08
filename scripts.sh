python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                    --opt shallow_cma \
                                    --pgd_test \
                                    --initial_prompt_text "a photo of a" \
                                    --learned_prompt_pos "prefix"
python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                    --opt shallow_cma \
                                    --pgd_test \
                                    --initial_prompt_text "a photo of a" \
                                    --learned_prompt_pos "middle"
python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                    --opt shallow_cma \
                                    --pgd_test \
                                    --initial_prompt_text "a photo of a" \
                                    --learned_prompt_pos "suffix"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 0.5 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "prefix"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 0.5 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "middle"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 0.5 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "suffix"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 1.0 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "prefix"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 1.0 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "middle"
# python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
#                                     --opt shallow_cma \
#                                     --adv_train \
#                                     --adv_train_all \
#                                     --adv_train_sample_ratio 1.0 \
#                                     --pgd_test \
#                                     --initial_prompt_text "a photo of a" \
#                                     --learned_prompt_pos "suffix"
