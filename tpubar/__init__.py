import os
import sys
import re
import time
import psutil
import platform
from subprocess import check_output

from tqdm.auto import tqdm
from threading import Thread

from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.python.profiler import profiler_client
#from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.framework import errors


def get_workers_list(cluster_resolver):
    worker_job_name = 'worker'
    cluster_spec = cluster_resolver.cluster_spec()
    if not cluster_spec:
        raise errors.UnavailableError(
            'None', 'None',
            'Cluster spec not found, your client must run in GCE environment.')
    task_indices = cluster_spec.task_indices(worker_job_name)
    workers_list = [
        cluster_spec.task_address(worker_job_name, i).replace(':8470', ':8466')
        for i in task_indices
    ]
    return ','.join(workers_list)


def FormatSize(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def run_command(cmd):
    out = check_output(cmd, shell=True)
    if isinstance(out, bytes):
        out = out.decode('utf8')
    return out

def queryhw():
    host_os = platform.system()
    if host_os == 'Linux':
        cpu_name = run_command("lscpu |grep 'Model name'")
        cpu_name = cpu_name.split(':')[-1].strip()
    elif host_os == 'Darwin':
        # dunno why a TPU would run on macos but i kept it anyways
        cpu_name = run_command("sysctl -n machdep.cpu.brand_string | sed -e 's/ *$//'").strip()
    else:
        cpu_name = platform.processor()

    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    return {'name': cpu_name, 'cores': cores, 'threads': threads}


class TPUMonitor:
    def __init__(self, tpu_name=None, refresh_secs=10, fileout=None, verbose=True, tpu_util='green', tpu_active='yellow', cpu_util='blue', ram_util='blue'):
        tpu_name = tpu_name if tpu_name else os.environ.get('TPU_NAME', None)
        tpu_cluster_resolver = resolver.TPUClusterResolver(tpu_name)
        service_addr = tpu_cluster_resolver.get_master()
        self.service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')
        self.workers_list = get_workers_list(tpu_cluster_resolver)
        self.monitoring_level = 2
        self.duration_ms = 1000
        self.refresh_secs = refresh_secs
        self.fileout = fileout if fileout else sys.stdout
        self.verbose = verbose
        self.colors = {
            'tpu_util': tpu_util,
            'tpu_active': tpu_active,
            'cpu_util': cpu_util,
            'ram_util': ram_util
        }
        self.current_stats = {}
        self.idx = 0
        self.hwdata()

    def start(self, daemon=True):
        _tpubarformat = f'TPU {self.mesh} Matrix Units: ' + '{bar} {percentage:.02f}% Utilization'
        _activebarformat = f'TPU {self.mesh} Active Time: ' + '{bar} {percentage:.02f}% Utilization'
        _cpubarformat = f'CPU {self.cpu}: ' + '{bar} {percentage:.02f}% Utilization'
        _rambarformat = 'RAM {desc} {bar} {percentage:.02f}% Utilization'
        self.tbar = tqdm(range(100), colour=self.colors['tpu_util'], bar_format=_tpubarformat, position=0, dynamic_ncols=True, leave=False, file=self.fileout)
        self.abar = tqdm(range(100), colour=self.colors['tpu_active'], bar_format=_activebarformat, position=1, dynamic_ncols=True, leave=False, file=self.fileout)
        self.cbar = tqdm(range(100), colour=self.colors['cpu_util'], bar_format=_cpubarformat, position=2, dynamic_ncols=True, leave=False, file=self.fileout)
        self.rbar = tqdm(range(100), colour=self.colors['ram_util'], bar_format=_rambarformat, position=3, dynamic_ncols=True, leave=False, file=self.fileout)
        if daemon:
            self.alive = True
            _background = Thread(target=self.background,)
            _background.start()
    
    def update(self):
        if (self.idx+1) % 10 == 0:
            self.clearbars()
        tpu_stats = self.tpu_util()
        if tpu_stats['mxu']:
            self.tbar.n = tpu_stats['mxu']
            self.tbar.refresh()

        self.cbar.n = self.cpu_utilization()
        self.cbar.refresh()
        rperc, rutil = self.ram_utilization()
        self.rbar.n = rperc
        self.rbar.set_description(rutil)
        self.rbar.refresh()
        idle_time = tpu_stats.get('idle_time', None)
        if idle_time:
            self.abar.n = (100.00 - idle_time)
            self.abar.refresh()
        self.current_stats = tpu_stats
        self.idx += 1

    def log(self, message):
        message = message + '\n' + ('------' * 25)
        self.tbar.write(message)

    def tpu_util(self):
        stats = {}
        util = self.tpu_utilization(self.service_addr, self.duration_ms, self.monitoring_level)
        util = util.split('\n')
        for stat in util:
            if 'TPU idle time' in stat:
                stats['idle_time'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['idle_string'] = stat.split('  ')[-1].strip()
            elif 'Utilization of TPU' in stat:
                stats['mxu'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['mxu_string'] = ' ' + stat.strip()
        return stats

    def background(self):
        while self.alive:
            try:
                self.update()
                time.sleep(self.refresh_secs)
            except:
                if self.verbose:
                    self.log('Another TPU Profiler is Active. Pausing')
                time.sleep(90)
                pass
            if not self.alive:
                break

    def hwdata(self):
        cpu_data = queryhw()
        self.cpu = cpu_data['name'].replace('CPU', '').strip() + ' ' + str(cpu_data['cores']) + 'vCPU/' + str(cpu_data['threads']) + ' Threads'
        util = self.tpu_utilization(self.service_addr, self.duration_ms, self.monitoring_level)
        util = util.split('\n')
        mesh_type = {'v': None, 'cores': None}
        for stat in util:
            if 'TPU type' in stat:
                tpu = stat.replace('TPU type: TPU', '').strip()
                mesh_type['v'] = tpu
            elif 'Number of TPU cores' in stat:
                idx = stat.find('(')
                mesh_cores = stat[:idx].strip()
                mesh_type['cores'] = re.search(r'[0-9]', mesh_cores).group()
        
        self.mesh = f'v{mesh_type["v"]}-{mesh_type["cores"]}'


    @classmethod
    def tpu_utilization(cls, service_addr, duration_ms, monitoring_level):
        return profiler_client.monitor(service_addr, duration_ms, monitoring_level)
    
    @classmethod
    def cpu_utilization(cls):
        return psutil.cpu_percent()
    
    @classmethod
    def ram_utilization(cls):
        ram = psutil.virtual_memory()
        rutil = f'{FormatSize(ram.used)}/{FormatSize(ram.total)}'
        return ram.percent, rutil
    
    def clearbars(self):
        self.tbar.clear()
        self.abar.clear()
        self.cbar.clear()
        self.rbar.clear()
    
    def closebars(self):
        self.alive = False
        self.tbar.close()
        self.abar.close()
        self.cbar.close()
        self.rbar.close()

    def __exit__(self, *_):
        self.closebars()
