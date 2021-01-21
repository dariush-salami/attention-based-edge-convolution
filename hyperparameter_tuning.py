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
GPU_CHECK_INTERVAL = 5 * 60
GPU_AVAILABLE_THRESHOLD = 10000
LOG_PATH_TEMPLATE = join(
    pathlib.Path(__file__).parent.absolute(),
    'logs/hyperparameter_tuning/gcn_layers_{gcn_layers}_st_factor_{st_factor}_k_{k}'
)
COMMAND_TEMPLATE = 'python train.py --t=1000 --gpu_id={gpu_id} --log_dir={log_dir} --k={k} ' \
                   '--spatio_temporal_factor={spatio_temporal_factor} ' \
                   '--graph_convolution_layers={graph_convolution_layers} '
HYPER_PARAMETERS = {
    'KS': [2, 4, 8, 16, 32],
    'ST_FACTORS': [0, 0.01, 0.05, 0.1, 10],
    'GCN_LAYERS': [1, 2, 3]
}
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
    global DONE_ARRAY, HYPER_PARAMETERS, COMMAND_TEMPLATE, LOG_PATH_TEMPLATE
    for GCN_LAYER in HYPER_PARAMETERS['GCN_LAYERS']:
        for ST_FACTOR in HYPER_PARAMETERS['ST_FACTORS']:
            for K in HYPER_PARAMETERS['KS']:
                if DONE_ARRAY is not None and len(DONE_ARRAY[(DONE_ARRAY[:, 0] == GCN_LAYER) & (
                        DONE_ARRAY[:, 1] == ST_FACTOR) & (DONE_ARRAY[:, 2] == K)]) > 0:
                    continue
                else:
                    if DONE_ARRAY is None:
                        DONE_ARRAY = np.array([[GCN_LAYER, ST_FACTOR, K, True]])
                    else:
                        DONE_ARRAY = np.vstack((DONE_ARRAY, np.array([GCN_LAYER, ST_FACTOR, K, True])))
                    log_string('Starting job for gcn_layers={}, st_factor={}, k={} on GPU={}'.format(GCN_LAYER,
                                                                                                ST_FACTOR,
                                                                                                K,
                                                                                                gpu_id))
                    log_dir = LOG_PATH_TEMPLATE.format(
                        gcn_layers=GCN_LAYER,
                        st_factor=ST_FACTOR,
                        k=K
                    )
                    command = COMMAND_TEMPLATE.format(
                        gpu_id=gpu_id,
                        log_dir=log_dir,
                        graph_convolution_layers=GCN_LAYER,
                        spatio_temporal_factor=ST_FACTOR,
                        k=K
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
    if current_running_jobs > MAXIMUM_CONCURRENT_JOBS:
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
