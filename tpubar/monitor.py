import os
import sys
import time
import re
import psutil
import tensorflow as tf

from tqdm.auto import tqdm
from threading import Thread, Lock
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver

from tpubar import env
from tpubar.host import queryhw
from tpubar.utils import FormatSize
from tpubar.network import TimeSeriesMonitor, get_workers_list, tpunicorn_query


if env['profiler']:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.python.profiler import profiler_v2 as profiler


_mesh_memory = {
    'v2-8': 6.872e+10,
    'v2-32': 2.749e+11,
    'v2-128': 1e+12,
    'v2-256': 2e+12,
    'v2-512': 4e+12,
    'v3-8': 1.374e+11,
    'v3-32': 5.498e+11,
    'v3-64': 1e+12,
    'v3-128': 2e+12,
    'v3-256': 4e+12,
    'v3-512': 8e+12
}


_timer_formats = {
    'secs': ['sec', 'secs', 'second', 'seconds', 's'],
    'mins': ['min', 'mins', 'minute', 'minutes', 'm'],
    'hrs': ['hr', 'hrs', 'hour', 'hours', 'h'],
    'days': ['day', 'days', 'd'],
    'wks': ['wk', 'wks', 'week', 'weeks' 'w']
}



class TPUMonitor:
    def __init__(self, tpu_name=None, project=None, profiler='v1', refresh_secs=10, fileout=None, verbose=False, disable=False, tpu_util='green', tpu_secondary='yellow', cpu_util='blue', ram_util='blue'):
        if profiler == 'trace':
            self.tpu_init_tf2(tpu_name)
        elif profiler in ['v1', 'v2']:
            self.tpu_init_tf2(tpu_name) if profiler == 'v2' else self.tpu_init_tf1(tpu_name, project)
        elif env['profiler'] or env['colab']:
            self.tpu_init_tf2(tpu_name)
        else:
            self.tpu_init_tf1(tpu_name, project)

        self.refresh_secs = refresh_secs
        self.fileout = fileout if fileout else sys.stdout
        self.verbose = verbose
        self.bars_disabled = disable
        self.colors = {
            'tpu_util': tpu_util,
            'tpu_secondary': tpu_secondary,
            'cpu_util': cpu_util,
            'ram_util': ram_util
        }
        self.current_stats = {}
        self.hooks = {}
        self.idx = 0
        self.hwdata()
        self.time = time.time()
        self._lock = Lock()

    def start(self, daemon=True):
        _tpubarformat = f'TPU {self.mesh} Matrix Units: ' + '{bar} {percentage:.02f}% Utilization'
        if self.profiler_ver == 'v2':
            _tpusecondarybarformat = f'TPU {self.mesh} Active Time: ' + '{bar} {percentage:.02f}% Utilization'  
        else:
            _tpusecondarybarformat = f'TPU {self.mesh} Memory: ' + '{desc} {bar} {percentage:.02f}% Utilization'  
        _cpubarformat = f'CPU {self.cpu}: ' + '{bar} {percentage:.02f}% Utilization'
        _rambarformat = 'RAM {desc} {bar} {percentage:.02f}% Utilization'
        self.tbar = tqdm(range(100), colour=self.colors['tpu_util'], bar_format=_tpubarformat, position=0, dynamic_ncols=True, leave=True, file=self.fileout, disable=self.bars_disabled)
        self.t2bar = tqdm(range(100), colour=self.colors['tpu_secondary'], bar_format=_tpusecondarybarformat, position=1, dynamic_ncols=True, leave=True, file=self.fileout, disable=self.bars_disabled)
        self.cbar = tqdm(range(100), colour=self.colors['cpu_util'], bar_format=_cpubarformat, position=2, dynamic_ncols=True, leave=True, file=self.fileout, disable=self.bars_disabled)
        self.rbar = tqdm(range(100), colour=self.colors['ram_util'], bar_format=_rambarformat, position=3, dynamic_ncols=True, leave=True, file=self.fileout, disable=self.bars_disabled)
        if daemon:
            self.alive = True
            _background = Thread(target=self.background, daemon=True)
            _background.start()
    
    def update(self):
        self.idx += 1
        tpu_stats = self.tpu_profiler()
        if tpu_stats['tpu_mxu']:
            self.tbar.n = tpu_stats['tpu_mxu']
            
        if self.profiler_ver == 'v2':
            idle_time = tpu_stats.get('tpu_idle_time', None)
            if idle_time:
                self.t2bar.n = (100.00 - idle_time)

        else:
            tpu_mem = tpu_stats.get('tpu_mem_per', None)
            if tpu_mem:
                self.t2bar.n = tpu_mem
                self.t2bar.set_description(tpu_stats.get('tpu_mem_str', ''), refresh=False)
        
        cpu_util = self.cpu_utilization()
        self.cbar.n = cpu_util
        rperc, rutil, rutilstr = self.ram_utilization()
        tpu_stats.update({'cpu_util': cpu_util, 'ram_per': rperc, 'ram_util': rutil, 'ram_util_str': rutilstr})
        self.rbar.n = rperc
        self.rbar.set_description(rutilstr, refresh=False)
        self.current_stats = tpu_stats
        self.refresh_all()
        self.fire_hooks(str(tpu_stats))

    def refresh_all(self):
        if (self.idx+1) % 10 == 0:
            self.clearbars()
        self.tbar.refresh()
        self.t2bar.refresh()
        self.cbar.refresh()
        self.rbar.refresh()

    def log(self, message):
        if not isinstance(message, str):
            message = str(message)
        message = message + '\n' + ('------' * 25)
        self.tbar.write(message)

    def background(self):
        while self.alive:
            with self._lock:
                try:
                    self.update()
                    time.sleep(self.refresh_secs)
                
                except KeyboardInterrupt:
                    self.log('Exiting')
                    self.alive = False
                    self.closebars()
                
                except Exception as e:
                    if self.verbose:
                        self.log(f'Another TPU Profiler is Active. Pausing. Error: {str(e)}')
                    time.sleep(60)
                    pass
                if not self.alive:
                    break

    def hwdata(self):
        cpu_data = queryhw()
        self.cpu = cpu_data['name'].replace('CPU', '').strip() + ' ' + str(cpu_data['cores']) + 'vCPU/' + str(cpu_data['threads']) + ' Threads'

    def tpu_util(self):
        stats = {'tpu_idle_time': 100.00, 'tpu_idle_str': '', 'tpu_mxu': 0.00, 'tpu_mxu_str': ''}
        util = self.tpu_utilization(self.service_addr, self.duration_ms, self.monitoring_level)
        util = util.split('\n')
        for stat in util:
            if 'TPU idle time' in stat:
                stats['tpu_idle_time'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['tpu_idle_str'] = stat.split('  ')[-1].strip()
            elif 'Utilization of TPU' in stat:
                stats['tpu_mxu'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['tpu_mxu_str'] = ' ' + stat.strip()
        return stats
    
    def tpu_api(self):
        mxu = self.monitor('tpu_core_mxu')
        for x, lst in mxu.items():
            curr_mxu = lst[0][-1]
        mem = self.monitor('tpu_container_mem')
        for x, lst in mem.items():
            curr_mem = lst[0][-1]
        mem_used, mem_str = FormatSize(curr_mem)
        vm_cpu = self.monitor('vm_cpu')
        for x, lst in vm_cpu.items():
            curr_cpu = lst[0][-1]
        tpu_cpu = self.monitor('tpu_host_cpu')
        for x, lst in tpu_cpu.items():
            curr_tpucpu = lst[0][-1]
        if self.tpu_max_mem <= curr_mem:
            self.tpu_max_mem = curr_mem + 1e+9
        mem_perc = curr_mem / self.tpu_max_mem
        _, total_mem_str = FormatSize(self.tpu_max_mem)
        
        stats = {
            'tpu_mxu': curr_mxu,
            'tpu_mem_per': (mem_perc * 100),
            'tpu_mem_util': mem_used,
            'tpu_mem_str': f'{mem_str}/{total_mem_str}',
            'tpu_vm_cpu_per': curr_cpu,
            'tpu_host_cpu_per': curr_tpucpu,
        }
        return stats

    def tpu_init_tf1(self, tpu_name=None, project=None):
        if tpu_name:
            os.environ['TPU_NAME'] = tpu_name
        tpu_config = tpunicorn_query(project)
        self.monitor = TimeSeriesMonitor(project_id=project)
        self.mesh = tpu_config['mesh']
        self.tpu_max_mem = _mesh_memory[self.mesh]
        self.profiler_ver = 'v1'
        self.tpu_profiler = self.tpu_api

    def tpu_init_tf2(self, tpu_name=None):
        tpu_name = tpu_name if tpu_name else os.environ.get('TPU_NAME', None)
        tpu_cluster_resolver = resolver.TPUClusterResolver(tpu_name)
        service_addr = tpu_cluster_resolver.get_master()
        self.service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')
        self.workers_list = get_workers_list(tpu_cluster_resolver)
        self.monitoring_level = 2
        self.duration_ms = 1000
        util = self.tpu_utilization(self.service_addr, self.duration_ms, self.monitoring_level)
        util = util.split('\n')
        mesh_type = {'v': 'v2', 'cores': 8}
        for stat in util:
            if 'TPU type' in stat:
                tpu = stat.replace('TPU type: TPU', '').strip()
                mesh_type['v'] = tpu
            elif 'Number of TPU cores' in stat:
                idx = stat.find('(')
                mesh_cores = stat[:idx].strip()
                mesh_type['cores'] = re.search(r'[0-9]', mesh_cores).group()
        
        self.mesh = f'{mesh_type["v"]}-{mesh_type["cores"]}'
        self.tpu_max_mem = _mesh_memory[self.mesh]
        self.profiler_ver = 'v2'
        self.tpu_profiler = self.tpu_util

    def get_time(self, fmt='mins'):
        _stoptime = time.time()
        total_time = _stoptime - self.time
        if fmt in _timer_formats['wks']:
            total_time /= 604800
        elif fmt in _timer_formats['days']:
            total_time /= 86400
        elif fmt in _timer_formats['hrs']:
            total_time /= 3600
        elif fmt in _timer_formats['mins']:
            total_time /= 60
        return total_time

    def add_hook(self, name, hook, freq=10):
        self.hooks[name] = {'freq': freq, 'function': hook}
        self.log(f'Added new hook {name}. Will call hook once every {freq} updates.')

    def rm_hook(self, name):
        if self.hooks.get(name, None):
            hook = self.hooks.pop(name)
            self.log(f'Removing hook {name}')
        else:
            self.log(f'Hook {name} not found')

    def fire_hooks(self, message, *args, **kwargs):
        if self.verbose:
            self.log(message)
        if self.hooks:
            for hook_name in self.hooks:
                hook = self.hooks[hook_name]
                if self.idx % hook['freq'] == 0:
                    hook['func'](message, *args, **kwargs)


    @classmethod
    def tpu_utilization(cls, service_addr, duration_ms, monitoring_level):
        return profiler_client.monitor(service_addr, duration_ms, monitoring_level)
    
    @classmethod
    def cpu_utilization(cls):
        return psutil.cpu_percent()
    
    @classmethod
    def ram_utilization(cls):
        ram = psutil.virtual_memory()
        rused, rusedstr = FormatSize(ram.used)
        _, rtotal = FormatSize(ram.total)
        rutil = f'{rusedstr}/{rtotal}'
        return ram.percent, rused, rutil
    
    def trace(self):
        self.trace_dir = os.path.join(env['dir'], 'logs')
        os.makedirs(self.trace_dir, exist_ok=True)
        options = profiler.ProfilerOptions(host_tracer_level=self.monitoring_level)
        while self.alive:
            with self._lock:
                try:
                    profiler_client.trace(self.service_addr, self.trace_dir, self.duration_ms, self.workers_list, 5, options)
                except KeyboardInterrupt:
                    self.alive = False
                    print('Closing Tracer')
                    sys.exit()
    
    def clearbars(self):
        self.tbar.clear()
        self.t2bar.clear()
        self.cbar.clear()
        self.rbar.clear()
    
    def close(self, *_):
        self.closebars()

    def closebars(self):
        self.alive = False
        self.tbar.close()
        self.t2bar.close()
        self.cbar.close()
        self.rbar.close()

    def __exit__(self, *_):
        self.closebars()
    
    def __enter__(self):
        return self