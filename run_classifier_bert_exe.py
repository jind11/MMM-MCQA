import sys
import os

task_name = sys.argv[1]
BERT_DIR = sys.argv[2]
per_gpu_train_batch_size = sys.argv[3]
gradient_accumulation_steps = sys.argv[4]

# set up some hypers
if task_name == 'dream':
    learning_rate = 2e-5
    num_train_epochs = 8.
    max_grad_norm = 1.
elif task_name == 'dream,race':
    learning_rate = 2e-5
    num_train_epochs = 5
    max_grad_norm = 1.
elif task_name in ['mctest160,race', 'mctest500,race']:
    learning_rate = 1.5e-5
    num_train_epochs = 5
    max_grad_norm = 0.
elif task_name in ['mctest160', 'mctest500'] or 'mcscript' in task_name:
    learning_rate = 1e-5
    num_train_epochs = 8.
    max_grad_norm = 0.
elif 'toefl' in task_name:
    learning_rate = 1e-5
    num_train_epochs = 10
    max_grad_norm = 0.
elif 'race' in task_name:
    learning_rate = 3e-5
    num_train_epochs = 5
    max_grad_norm = 0.
elif 'nli' in task_name:
    learning_rate = 2e-5
    num_train_epochs = 4.
    max_grad_norm = 0.
else:
    raise NotImplementedError

# set up data directories
data_dir = []
for task_name_ in task_name.split(','):
    if task_name_ == 'race':
        data_dir.append('data/RACE')
    elif task_name_ == 'dream':
        data_dir.append('data/dream')
    elif task_name_ in ['toefl', 'mctest', 'mcscript']:
        data_dir.append('data/{}'.format(task_name_.upper()))
    elif task_name_ in ['mctest160', 'mctest500']:
        data_dir.append('data/{}'.format(task_name_.upper()[:6]))
    else:
        data_dir.append('data/{}'.format(task_name_.upper()))
data_dir = ','.join(data_dir)

# start running the model
command = 'python run_classifier_bert.py --task_name {task_name} --do_train ' \
          '--do_eval ' \
          '--data_dir {data_dir} ' \
          '--bert_model {BERT_DIR} ' \
          '--per_gpu_train_batch_size {per_gpu_train_batch_size} ' \
          '--do_lower_case ' \
          '--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
          '--seed 1 --max_grad_norm {max_grad_norm} ' \
          '--output_dir tmp/{task_name}_{BERT_DIR} ' \
          '--gradient_accumulation_steps {gradient_accumulation_steps} '.format(
                                                    BERT_DIR=BERT_DIR,
                                                    data_dir=data_dir, task_name=task_name,
                                                    per_gpu_train_batch_size=per_gpu_train_batch_size,
                                                    learning_rate=learning_rate,
                                                    num_train_epochs=num_train_epochs,
                                                    max_grad_norm=max_grad_norm,
                                                    gradient_accumulation_steps=gradient_accumulation_steps)

os.system(command)