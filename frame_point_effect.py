import sched
import subprocess
import time
from io import StringIO
import numpy as np
import pandas as pd
from os.path import join
import pathlib
import sys

MAXIMUM_CONCURRENT_JOBS = 3
PROCESS_LIST = []
GPU_CHECK_INTERVAL = 2 * 60
GPU_AVAILABLE_THRESHOLD = 10000
LOG_PATH_TEMPLATE = join(
    pathlib.Path(__file__).parent.absolute(),
    'logs/secon/pantomime/frame_point_effect/{frame}_frame/{points}_points'
)
TRAIN_OR_EVAL = 'TRAIN'
TRAIN_TEMPLATE = 'python train.py --t=1000 --gpu_id={gpu_id} --log_dir={log_dir} --k={k} ' \
                 '--spatio_temporal_factor=10 ' \
                 '--graph_convolution_layers=1 ' \
                 '--dataset={dataset}'

EVALUATE_TEMPLATE = 'python evaluate.py --t=1000 --gpu_id={gpu_id} --log_dir={log_dir} --k={k} ' \
                    '--spatio_temporal_factor=10  ' \
                    '--graph_convolution_layers=1 ' \
                    '--eval_score_path={eval_score_path} ' \
                    '--dataset={dataset}'

DATASETS = [
    {'path': 'data/pantomime/without_outlier_removal/2_frame/512_points', 'k': 32, 'frame': 2, 'points': 512},
    {'path': 'data/pantomime/without_outlier_removal/4_frame/256_points', 'k': 32, 'frame': 4, 'points': 256},
    {'path': 'data/pantomime/without_outlier_removal/8_frame/128_points', 'k': 32, 'frame': 8, 'points': 128},
    {'path': 'data/pantomime/without_outlier_removal/16_frame/64_points', 'k': 32, 'frame': 16, 'points': 64},
    {'path': 'data/pantomime/without_outlier_removal/64_frame/16_points', 'k': 16, 'frame': 64, 'points': 16},
]
# GCN_LAYER, ST_FACTOR, K
DONE_ARRAY = None
AVAILABLE_GPU_QUERY = 'nvidia-smi --query-gpu=index,memory.free --format=csv'

LOG_FOUT = open('hyper_parameter_tuning_log.txt', 'w')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
    sys.stdout.flush()


def start_next_job_on_gpu(gpu_id):
    global DONE_ARRAY, DATASETS, TRAIN_TEMPLATE, EVALUATE_TEMPLATE, LOG_PATH_TEMPLATE, TRAIN_OR_EVAL
    for DATASET in DATASETS:
        if DONE_ARRAY is not None and len(DONE_ARRAY[DONE_ARRAY[:, 0] == DATASET['path']]) > 0:
            continue
        else:
            if DONE_ARRAY is None:
                DONE_ARRAY = np.array([[DATASET['path'], True]])
            else:
                DONE_ARRAY = np.vstack((DONE_ARRAY, np.array([DATASET['path'], True])))
            log_string('Starting job for dataset={} on GPU={}'.format(DATASET['path'],
                                                                      gpu_id))
            log_dir = LOG_PATH_TEMPLATE.format(
                k=DATASET['k'],
                frame=DATASET['frame'],
                points=DATASET['points']
            )
            if TRAIN_OR_EVAL == 'TRAIN':
                command = TRAIN_TEMPLATE.format(
                    gpu_id=gpu_id,
                    log_dir=log_dir,
                    dataset=DATASET['path'],
                    k=DATASET['k']
                )
            else:
                command = EVALUATE_TEMPLATE.format(
                    gpu_id=gpu_id,
                    log_dir=log_dir,
                    k=DATASET['k'],
                    dataset=DATASET['path'],
                    eval_score_path=log_dir
                )
            PROCESS_LIST.append(subprocess.Popen(command, shell=True))
            log_string(command)
            return True
    return False


def check_available_gpu(scheduler):
    global AVAILABLE_GPU_QUERY, GPU_CHECK_INTERVAL, GPU_AVAILABLE_THRESHOLD, PROCESS_LIST, MAXIMUM_CONCURRENT_JOBS
    log_string('Checking concurrent jobs criterion.')
    current_running_jobs = 0
    finished_processes_indices = []
    for p_index, process in enumerate(PROCESS_LIST):
        poll = process.poll()
        if poll is None:
            current_running_jobs += 1
        else:
            finished_processes_indices.append(p_index)
    for index in sorted(finished_processes_indices, reverse=True):
        del PROCESS_LIST[index]
    if current_running_jobs >= MAXIMUM_CONCURRENT_JOBS:
        log_string('There are {}/{} running jobs. We should wait for one to finish first!'.format(
            MAXIMUM_CONCURRENT_JOBS, MAXIMUM_CONCURRENT_JOBS
        ))
        scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))
        return
    else:
        log_string('There are {}/{} running jobs. Let\'s lunch a new one!!'.format(
            current_running_jobs, MAXIMUM_CONCURRENT_JOBS
        ))
    log_string('GPU availability check...')
    result = subprocess.run(AVAILABLE_GPU_QUERY, stdout=subprocess.PIPE, shell=True)
    std_out_array = StringIO(result.stdout.decode('utf-8'))
    gpu_info = pd.read_csv(std_out_array)
    gpu_info.columns = ['gpu_id', 'free_memory']
    gpu_info['gpu_id'] = gpu_info['gpu_id'].astype(int)
    gpu_info['free_memory'] = gpu_info['free_memory'].str.extract(r'(\d+)', expand=False).astype(float)
    available_gpu = gpu_info[gpu_info['free_memory'] >= GPU_AVAILABLE_THRESHOLD]
    restart_scheduler = False
    if available_gpu.empty:
        log_string('There is no GPU available at the moment.')
        restart_scheduler = True
    else:
        log_string('{} GPU(s) is(are) available for the next job.'.format(len(available_gpu.index)))
        if not start_next_job_on_gpu(available_gpu.loc[available_gpu.index[0], 'gpu_id']):
            log_string('There is no job to run! Terminating the program!')
        else:
            restart_scheduler = True
    if restart_scheduler:
        scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))


scheduler = sched.scheduler(time.time, time.sleep)
scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))
scheduler.run()
