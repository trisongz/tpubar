import os
import sys
import re
import time
import psutil
import platform
import collections
import calendar
import simdjson as json

import tensorflow as tf

from subprocess import check_output
from tqdm.auto import tqdm
from threading import Thread, Lock
from datetime import datetime

parser = json.Parser()
env = dict()

try:
    import google.colab
    env['colab'] = True
except ImportError:
    env['colab'] = False


env['tf2'] = True if tf.__version__.startswith('2') else False
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
try:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.python.framework import errors
    env['profiler'] = True
except ImportError:
    env['profiler'] = False

import google.auth
from google.cloud import monitoring_v3
from google.protobuf.json_format import MessageToJson

# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def pb_to_json(pb):
    """Converts arbitrary protobuf messages into JSON"""
    return MessageToJson(pb)


def pb_to_dict(pb):
    """Converts arbitrary protobuf messages into python dicts"""
    return parser.parse(pb_to_json(pb)).as_dict()

def utc():
    d = datetime.utcnow()
    return calendar.timegm(d.utctimetuple())


metrics = {
    'vm_cpu': "compute.googleapis.com/instance/cpu/utilization",
    'vm_net_sent': "compute.googleapis.com/instance/network/sent_bytes_count",
    'vm_net_recv': "compute.googleapis.com/instance/network/received_bytes_count",
    'vm_disk_write': "compute.googleapis.com/instance/disk/write_bytes_count",
    'vm_disk_read': "compute.googleapis.com/instance/disk/read_bytes_count",
    'tpu_core_mxu': "tpu.googleapis.com/tpu/mxu/utilization",
    'tpu_container_cpu': "tpu.googleapis.com/container/cpu/utilization",
    'tpu_container_mem': "tpu.googleapis.com/container/memory/usage",
    'tpu_host_cpu': "tpu.googleapis.com/cpu/utilization",
    'tpu_host_mem': "tpu.googleapis.com/memory/usage",
    'tpu_host_net_sent': "tpu.googleapis.com/network/sent_bytes_count",
    'tpu_host_net_recv': "tpu.googleapis.com/network/received_bytes_count",
}

def gce_series_info(series):
    h = {k: pb_to_dict(getattr(series, k)) for k in "metric resource metadata".split()}
    h = {k: v for k, v in h.items() if len(v) > 0}
    return flatten(h)


def gce_instance_labeler(series, **options):
    if options.get('short'):
        return series.metric.labels['instance_name']
    r = []
    r += [k+'/'+series.resource.labels[k] for k in 'project_id zone'.split()]
    r += [k+'/'+series.metric.labels[k] for k in 'instance_name'.split()]
    return '/'.join(r)


def gce_instance_disk_labeler(series, **options):
    if options.get('short'):
        return '/'.join([series.metric.labels[k] for k in 'instance_name device_name'.split()])
    r = []
    r += [k+'/'+series.resource.labels[k] for k in 'project_id zone'.split()]
    r += [k+'/'+series.metric.labels[k] for k in 'instance_name device_name'.split()]
    return '/'.join(r)


def gce_series_getattrs(series, attrs, *, short=False):
    if isinstance(attrs, str):
        attrs = attrs.split()
    if short:
        r  = [series.resource.labels[k] for k in attrs if len(series.resource.labels[k]) > 0]
        r += [series.metric.labels[k] for k in attrs if len(series.metric.labels[k]) > 0]
    else:
        r  = [k+'/'+series.resource.labels[k] for k in attrs if len(series.resource.labels[k]) > 0]
        r += [k+'/'+series.metric.labels[k] for k in attrs if len(series.metric.labels[k]) > 0]
    return '/'.join(r)


def gce_tpu_labeler(series, **options):
    if options.get('short'):
        return gce_series_getattrs(series, 'node_id worker_id core container_name', short=True)
    return gce_series_getattrs(series, 'project_id zone node_id worker_id core container_name')


labelers = {
    "compute.googleapis.com/instance/network/sent_bytes_count":
    gce_instance_labeler,
    "compute.googleapis.com/instance/network/received_bytes_count":
    gce_instance_labeler,
    "compute.googleapis.com/instance/cpu/utilization":
    gce_instance_labeler,
    "compute.googleapis.com/instance/disk/write_bytes_count":
    gce_instance_disk_labeler,
    "compute.googleapis.com/instance/disk/read_bytes_count":
    gce_instance_disk_labeler,
    "tpu.googleapis.com/tpu/mxu/utilization":
    gce_tpu_labeler,
    "tpu.googleapis.com/container/memory/usage":
    gce_tpu_labeler,
    "tpu.googleapis.com/cpu/utilization":
    gce_tpu_labeler,
    "tpu.googleapis.com/memory/usage":
    gce_tpu_labeler,
    "tpu.googleapis.com/network/sent_bytes_count":
    gce_tpu_labeler,
    "tpu.googleapis.com/network/received_bytes_count":
    gce_tpu_labeler,
}


def get_time_series_label(ts, **options):
    return labelers[ts.metric.type](ts, **options)

def get_default_project_id():
    _, project_id = google.auth.default()
    return project_id

class TimeSeriesMonitor:
    def __init__(self, project_id=None, client=None):
        if project_id is None:
            project_id = get_default_project_id()
        self.project_id = project_id
        # [START monitoring_read_timeseries_simple]
        if client is None:
            client = monitoring_v3.MetricServiceClient()
        self.client = client

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def get(self, metric="tpu_mxu", node_id=None, interval=None, filters=None, raw=False, when=None, full_names=False):
        if when is None:
            when = utc()

        if '/' not in metric:
            metric = metrics[metric]

        if interval is None:
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": seconds, "nanos": nanos},
                    "start_time": {"seconds": (seconds - 1200), "nanos": nanos},
                }
            )

        if filters is None:
            filters = []
        filters = filters[:]
        if node_id is not None:
            filters += [['resource.labels.node_id', node_id]]
        filters += [['metric.type', metric]]
        filters = ' AND '.join(['{} = {}'.format(k, json.dumps(v)) for k, v in filters])

        results = self.client.list_time_series(
            request={
                "name": "projects/{project_id}".format(project_id=self.project_id),
                "filter": filters,
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
            }
        )
        if raw:
            return results
        points = collections.defaultdict(lambda: [])
        for timeSeries in results:
            key = get_time_series_label(timeSeries, short=not full_names)
        for point in timeSeries.points:
            point_utc = point.interval.start_time.timestamp()
            seconds_ago = int(when - point_utc)
            if timeSeries.value_type == 2: # what's the correct way to get INT64 here?
                value = point.value.int64_value
            else:
                value = point.value.double_value
            points[key].append([seconds_ago, value])
        points = dict(points)
        return points


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
            return bytes, f"{bytes:.2f}{unit}{suffix}"
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



def parse_tpu_data(tpu):
    data = tpu['name'].split('/')
    tpu_name, tpu_zone = data[-1], data[-3]
    tpu_config = {
        'name': tpu_name,
        'mesh': tpu['acceleratorType'],
        'region': tpu_zone,
        'master': tpu['ipAddress'] if 'ipAddress' in tpu else None,
    }
    return tpu_config

def tpunicorn_query(project):
    config = {'project': project}
    if not env['colab']:
        import tpunicorn
        tpu_data = tpunicorn.tpu.get_tpus(project=project)
        selected_tpu = None
        tpu_name = os.environ.get('TPU_NAME', None)
        if len(tpu_data) > 1:
            if tpu_name:
                for x, tpu in enumerate(tpu_data):
                    if tpu_name in tpu['name']:
                        selected_tpu = tpu_data[x]
            else:
                for x, tpu in enumerate(tpu_data):
                    print(f'[{x}] - {tpu}')
                
                tpu_idx = input('Select TPU')
                selected_tpu = tpu_data[tpu_idx]
            
        else:
            selected_tpu = tpu_data[0]

        tpu_config = parse_tpu_data(selected_tpu)
        config.update(tpu_config)
        
    else:
        config['master'] = os.environ['TPU_NAME']
        config['name'] = os.environ['TPU_NAME']
        config['region'] = 'us'
        config['mesh'] = 'v2-8'
    return config

_mesh_memory = {
    'v2-8': 6.872e+10,
    'v3-8': 1.374e+11,
}

class TPUMonitor:
    def __init__(self, tpu_name=None, project=None, profiler='v1', refresh_secs=10, fileout=None, verbose=True, tpu_util='green', tpu_secondary='yellow', cpu_util='blue', ram_util='blue'):
        if profiler in ['v1', 'v2']:
            self.tpu_init_tf2(tpu_name) if profiler == 'v2' else self.tpu_init_tf1(tpu_name)
        elif env['profiler'] or env['colab']:
            self.tpu_init_tf2(tpu_name)
        else:
            self.tpu_init_tf1(tpu_name, project)

        self.refresh_secs = refresh_secs
        self.fileout = fileout if fileout else sys.stdout
        self.verbose = verbose
        self.colors = {
            'tpu_util': tpu_util,
            'tpu_secondary': tpu_secondary,
            'cpu_util': cpu_util,
            'ram_util': ram_util
        }
        self.current_stats = {}
        self.idx = 0
        self.hwdata()
        self._lock = Lock()

    def start(self, daemon=True):
        _tpubarformat = f'TPU {self.mesh} Matrix Units: ' + '{bar} {percentage:.02f}% Utilization'
        if self.profiler_ver == 'v2':
            _tpusecondarybarformat = f'TPU {self.mesh} Active Time: ' + '{bar} {percentage:.02f}% Utilization'  
        else:
            _tpusecondarybarformat = f'TPU {self.mesh} Memory: ' + '{desc} {bar} {percentage:.02f}% Utilization'  
        _cpubarformat = f'CPU {self.cpu}: ' + '{bar} {percentage:.02f}% Utilization'
        _rambarformat = 'RAM {desc} {bar} {percentage:.02f}% Utilization'
        self.tbar = tqdm(range(100), colour=self.colors['tpu_util'], bar_format=_tpubarformat, position=0, dynamic_ncols=True, leave=True, file=self.fileout)
        self.t2bar = tqdm(range(100), colour=self.colors['tpu_secondary'], bar_format=_tpusecondarybarformat, position=1, dynamic_ncols=True, leave=True, file=self.fileout)
        self.cbar = tqdm(range(100), colour=self.colors['cpu_util'], bar_format=_cpubarformat, position=2, dynamic_ncols=True, leave=True, file=self.fileout)
        self.rbar = tqdm(range(100), colour=self.colors['ram_util'], bar_format=_rambarformat, position=3, dynamic_ncols=True, leave=True, file=self.fileout)
        if daemon:
            self.alive = True
            _background = Thread(target=self.background, daemon=True)
            _background.start()
    
    def update(self):
        tpu_stats = self.tpu_profiler()
        if tpu_stats['tpu_mxu']:
            self.tbar.n = tpu_stats['tpu_mxu']
            #self.tbar.refresh(nolock=True)
            
        if self.profiler_ver == 'v2':
            idle_time = tpu_stats.get('tpu_idle_time', None)
            if idle_time:
                self.t2bar.n = (100.00 - idle_time)
                #self.t2bar.refresh(nolock=True)

        else:
            tpu_mem = tpu_stats.get('tpu_memory_percent', None)
            if tpu_mem:
                self.t2bar.n = tpu_mem
                self.t2bar.set_description(tpu_stats.get('tpu_memory_string', ''), refresh=False)
                #self.t2bar.refresh(nolock=True)
                
        self.cbar.n = self.cpu_utilization()
        rperc, rutil = self.ram_utilization()
        self.rbar.n = rperc
        self.rbar.set_description(rutil, refresh=False)
        #self.rbar.refresh(nolock=True)
        self.current_stats = tpu_stats

        self.refresh_all()
        if self.verbose:
            self.log(str(tpu_stats))
        
        self.idx += 1

    def refresh_all(self):
        if (self.idx+1) % 2 == 0:
            self.clearbars()
        self.tbar.refresh()
        self.t2bar.refresh()
        self.cbar.refresh()
        self.rbar.refresh()


    def log(self, message):
        message = message + '\n' + ('------' * 25)
        self.tbar.write(message)

    def background(self):
        while self.alive:
            with self._lock:
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

    def tpu_util(self):
        stats = {}
        util = self.tpu_utilization(self.service_addr, self.duration_ms, self.monitoring_level)
        util = util.split('\n')
        for stat in util:
            if 'TPU idle time' in stat:
                stats['tpu_idle_time'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['tpu_idle_string'] = stat.split('  ')[-1].strip()
            elif 'Utilization of TPU' in stat:
                stats['tpu_mxu'] = float(stat.split(':')[-1].replace('%','').strip())
                stats['tpu_mxu_string'] = ' ' + stat.strip()
        return stats
    
    def tpu_api(self):
        mxu = self.monitor('tpu_core_mxu')
        for x, lst in mxu.items():
            curr_mxu = lst[0][-1]
        mem = self.monitor('tpu_container_mem')
        for x, lst in mem.items():
            curr_mem = lst[0][-1]
        mem_used, mem_str = FormatSize(curr_mem)
        mem_perc = curr_mem / _mesh_memory[self.mesh]
        _, total_mem_str = FormatSize(_mesh_memory[self.mesh])
        
        stats = {
            'tpu_mxu': curr_mxu,
            'tpu_memory_percent': min(mem_perc * 100, 100.00),
            'tpu_memory_used': mem_used,
            'tpu_memory_string': f'{mem_str}/{total_mem_str}'
        }
        return stats

    def tpu_init_tf1(self, tpu_name=None, project=None):
        if tpu_name:
            os.environ['TPU_NAME'] = tpu_name
        tpu_config = tpunicorn_query(project)
        self.monitor = TimeSeriesMonitor()
        self.mesh = tpu_config['mesh']
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
        self.profiler_ver = 'v2'
        self.tpu_profiler = self.tpu_util

    @classmethod
    def tpu_utilization(cls, service_addr, duration_ms, monitoring_level):
        return profiler_client.monitor(service_addr, duration_ms, monitoring_level)
    
    @classmethod
    def cpu_utilization(cls):
        return psutil.cpu_percent()
    
    @classmethod
    def ram_utilization(cls):
        ram = psutil.virtual_memory()
        _, rused = FormatSize(ram.used)
        _, rtotal = FormatSize(ram.total)
        rutil = f'{rused}/{rtotal}'
        return ram.percent, rutil
    
    def clearbars(self):
        self.tbar.clear()
        self.t2bar.clear()
        self.cbar.clear()
        self.rbar.clear()
    
    def closebars(self):
        self.alive = False
        self.tbar.close()
        self.t2bar.close()
        self.cbar.close()
        self.rbar.close()

    def __exit__(self, *_):
        self.closebars()