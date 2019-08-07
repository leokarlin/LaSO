'''
Example of how to use the LaSO model in your code.
'''

# import main from the appropriate script
from scripts_coco.train_setops_stripped import Main
# from scripts_coco.test_retrieval import Main
# from scripts_coco.test_precision import Main

# define an instance of the main class
main_instance = Main()
# define model paramters
main_instance.coco_path = '/dccstor/leonidka1/data/coco'
main_instance.epochs = 50
# initialize model
main_instance.initialize()
# run model - test/ train
main_instance.start()

# test output folder structure
'''
(kef) [amitalfa@dccxc203 initial_layers]$ ll /dccstor/alfassy/results/test_retrieval/0218_8977e29/681989/190428_081951/
total 101137
-rw-r--r-- 1 amitalfa users      113 Apr 28 08:19 cmdline.txt
-rw-r--r-- 1 amitalfa users     4684 Apr 28 08:19 config.py
-rw-r--r-- 1 amitalfa users   192518 Apr 28 08:19 git_diff.txt
-rw-r--r-- 1 amitalfa users 15919239 Apr 28 10:21 results_a_I_b.pkl
-rw-r--r-- 1 amitalfa users 16202039 Apr 28 10:59 results_a.pkl
-rw-r--r-- 1 amitalfa users  7936839 Apr 28 09:01 results_a_S_b.pkl
-rw-r--r-- 1 amitalfa users 11644839 Apr 28 09:41 results_a_U_b.pkl
-rw-r--r-- 1 amitalfa users 15919239 Apr 28 10:41 results_b_I_a.pkl
-rw-r--r-- 1 amitalfa users 16202039 Apr 28 11:18 results_b.pkl
-rw-r--r-- 1 amitalfa users  7840439 Apr 28 09:21 results_b_S_a.pkl
-rw-r--r-- 1 amitalfa users 11644839 Apr 28 10:01 results_b_U_a.pkl
-rw-r--r-- 1 amitalfa users     2001 Apr 28 11:18 script_log
'''

# test output script_log example
'''
2019-04-28 08:19:51,766 [MainThread  ] [INFO ]  Created results path: /dccstor/alfassy/results/test_retrieval/0218_8977e29/681989/190428_081951
2019-04-28 08:19:51,787 [MainThread  ] [INFO ]  Setup the models.
2019-04-28 08:19:51,787 [MainThread  ] [INFO ]  Inception3 model
2019-04-28 08:19:53,886 [MainThread  ] [INFO ]  Initialize inception model using Amit's networks.
2019-04-28 08:20:08,524 [MainThread  ] [INFO ]  Resuming the models.
2019-04-28 08:20:08,524 [MainThread  ] [INFO ]  using paper models
2019-04-28 08:20:12,248 [MainThread  ] [INFO ]  Setting up the datasets.
2019-04-28 08:20:12,315 [MainThread  ] [INFO ]  Copying data to tmp
2019-04-28 08:22:54,842 [MainThread  ] [INFO ]  Calculating indices.
2019-04-28 08:23:04,983 [MainThread  ] [INFO ]  Calculate the validation embeddings.
2019-04-28 08:29:16,558 [MainThread  ] [INFO ]  Calculate the embedding NN BallTree.
2019-04-28 08:29:23,840 [MainThread  ] [INFO ]  Calculate test set embedding.
2019-04-28 08:41:26,312 [MainThread  ] [INFO ]  Calculate scores.
2019-04-28 09:01:16,456 [MainThread  ] [INFO ]  Test a_S_b average recall (k=1, 3, 5): [0.15774941 0.30975018 0.39174374]
2019-04-28 09:21:36,746 [MainThread  ] [INFO ]  Test b_S_a average recall (k=1, 3, 5): [0.15835367 0.30721486 0.3891798 ]
2019-04-28 09:41:55,912 [MainThread  ] [INFO ]  Test a_U_b average recall (k=1, 3, 5): [0.62102854 0.72747891 0.76325716]
2019-04-28 10:01:54,786 [MainThread  ] [INFO ]  Test b_U_a average recall (k=1, 3, 5): [0.62122391 0.72644851 0.76149227]
2019-04-28 10:21:42,917 [MainThread  ] [INFO ]  Test a_I_b average recall (k=1, 3, 5): [0.67967224 0.78322097 0.81947849]
2019-04-28 10:41:30,038 [MainThread  ] [INFO ]  Test b_I_a average recall (k=1, 3, 5): [0.67914967 0.78431402 0.81867714]
2019-04-28 10:59:48,137 [MainThread  ] [INFO ]  Test a average recall (k=1, 3, 5): [0.61569753 0.72506626 0.76083887]
2019-04-28 11:18:05,401 [MainThread  ] [INFO ]  Test b average recall (k=1, 3, 5): [0.61773566 0.72654131 0.76237568]
'''

# train output folder structure
'''
(kef) [amitalfa@dccxc203 initial_layers]$ ll /dccstor/alfassy/results/train_setops_stripped/0218_8977e29/1650167/190507_101157/
total 1158881
-rw-r--r-- 1 amitalfa users       427 May  7 10:11 cmdline.txt
-rw-r--r-- 1 amitalfa users      6312 May  7 10:11 config.py
-rw-r--r-- 1 amitalfa users    262038 May  7 10:11 git_diff.txt
-rw------- 1 amitalfa users  94301854 May  7 15:20 networks_base_model_2.pth
-rw------- 1 amitalfa users  94301854 May  7 15:20 networks_base_model_2_val_acc=0.529.pth
-rw------- 1 amitalfa users  94301854 May  7 17:57 networks_base_model_3_val_acc=0.526.pth
-rw------- 1 amitalfa users  94301854 May  7 20:33 networks_base_model_4.pth
-rw------- 1 amitalfa users    656244 May  7 15:20 networks_classifier_2.pth
-rw------- 1 amitalfa users    656244 May  7 15:20 networks_classifier_2_val_acc=0.529.pth
-rw------- 1 amitalfa users    656244 May  7 17:57 networks_classifier_3_val_acc=0.526.pth
-rw------- 1 amitalfa users    656244 May  7 20:33 networks_classifier_4.pth
-rw------- 1 amitalfa users 201606120 May  7 15:20 networks_setops_model_2.pth
-rw------- 1 amitalfa users 201606120 May  7 15:20 networks_setops_model_2_val_acc=0.529.pth
-rw------- 1 amitalfa users 201606120 May  7 17:57 networks_setops_model_3_val_acc=0.526.pth
-rw------- 1 amitalfa users 201606120 May  7 20:33 networks_setops_model_4.pth
-rw-r--r-- 1 amitalfa users      5284 May  7 23:10 script_log
'''


