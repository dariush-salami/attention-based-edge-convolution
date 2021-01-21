import sched
import subprocess
import time
from io import StringIO
import numpy as np
import pandas as pd


GPU_CHECK_INTERVAL = 3
GPU_AVAILABLE_THRESHOLD = 10000
BASE_PATH = 'logs/hyperparameter_tuning/'
COMMAND_TEMPLATE = 'python train.py --t=1000 --gpu_id={} --log_dir={} --k={} --spatio_temporal_factor={} ' \
                   '--graph_convolution_layers={}} '
HYPER_PARAMETERS = {
    'KS': [2, 4, 8, 16, 32],
    'ST_FACTORS': [0, 0.01, 0.05, 0.1, 10],
    'GCN_LAYERS': [1, 2, 3]
}
# GCN_LAYER, ST_FACTOR, K
DONE_ARRAY = None
AVAILABLE_GPU_QUERY = 'nvidia-smi --query-gpu=index,memory.free --format=csv'


def start_next_job_on_gpu(gpu_id):
    global DONE_ARRAY, HYPER_PARAMETERS
    no_job_to_run = True
    for GCN_LAYER in HYPER_PARAMETERS['GCN_LAYERS']:
        for ST_FACTOR in HYPER_PARAMETERS['ST_FACTORS']:
            for K in HYPER_PARAMETERS['KS']:
                if DONE_ARRAY is not None and DONE_ARRAY[(DONE_ARRAY[:, 0] == GCN_LAYER) & (DONE_ARRAY[:, 1] == ST_FACTOR) & (DONE_ARRAY[:, 2] == K)]:
                    continue
                else:
                    no_job_to_run = False
                    if not DONE_ARRAY:
                        DONE_ARRAY = np.array([GCN_LAYER, ST_FACTOR, K, True])
                    else:
                        DONE_ARRAY = np.vstack((DONE_ARRAY, np.array([GCN_LAYER, ST_FACTOR, K, True])))
                    # TODO: start the job on GPU
    return no_job_to_run


def check_available_gpu(scheduler):
    global AVAILABLE_GPU_QUERY, GPU_CHECK_INTERVAL, GPU_AVAILABLE_THRESHOLD
    print('GPU availability check...')
    result = subprocess.run(AVAILABLE_GPU_QUERY, stdout=subprocess.PIPE, shell=True)
    std_out_array = StringIO(result.stdout.decode('utf-8'))
    gpu_info = pd.read_csv(std_out_array)
    gpu_info.columns = ['gpu_id', 'free_memory']
    gpu_info['gpu_id'] = gpu_info['gpu_id'].astype(float)
    gpu_info['free_memory'] = gpu_info['free_memory'].str.extract(r'(\d+)', expand=False).astype(float)
    available_gpu = gpu_info[gpu_info['free_memory'] >= GPU_AVAILABLE_THRESHOLD]
    restart_scheduler = False
    if available_gpu.empty:
        print('There is no GPU available at the moment.')
        restart_scheduler = True
    else:
        print('{} GPU(s) is(are) available for the next job.'.format(len(available_gpu.index)))
        if not start_next_job_on_gpu(available_gpu.loc[available_gpu.index[0], 'gpu_id']):
            print('There is no job to run! Terminating the program!')
        else:
            restart_scheduler = True
    if restart_scheduler:
        scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))


scheduler = sched.scheduler(time.time, time.sleep)
scheduler.enter(GPU_CHECK_INTERVAL, 1, check_available_gpu, (scheduler,))
scheduler.run()
