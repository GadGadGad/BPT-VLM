python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                     --opt shallow_cma \
                                     --pgd_test \
                                     --adv_train \
                                     --pgd_test_num_iter 20 \
                                     --adv_train_num_iter 20 \
                                     --parallel
python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                     --opt shallow_cma \
                                     --pgd_test \
                                     --pgd_test_num_iter 20 \
                                     --adv_train_num_iter 20 \
                                     --parallel
python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                     --opt shallow_cma \
                                     --pgd_test \
                                     --adv_train \
                                     --pgd_test_num_iter 10 \
                                     --adv_train_num_iter 20 \
                                     --parallel
python BBT_VL_Shallow_PGD.py --task_name CIFAR10 \
                                     --opt shallow_cma \
                                     --pgd_test \
                                     --pgd_test_num_iter 10 \
                                     --adv_train_num_iter 20 \
                                     --parallel