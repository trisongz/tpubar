
import os
import sys
import re
import calendar
import collections
import simdjson as json
import time

import google.auth

from datetime import datetime
from google.cloud import monitoring_v3
from google.protobuf.json_format import MessageToJson
from tpubar import env

if env['profiler']:
    from tensorflow.python.framework import errors

parser = json.Parser()

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
        elif project_id in ['tfork', 'tensorfork']:
            project_id = 'gpt-2-15b-poetry'
        self.project_id = project_id
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
    if project in ['tfork', 'tensorfork']:
        project = 'gpt-2-15b-poetry'
    config = {'project': project}
    if not env['colab']:
        import tpunicorn
        tpu_data = None
        for zone in ['europe-west4-a', 'us-central1-f', 'us-central1-a', 'us-central1-b', 'us-central1-c', 'asia-east1-c']:
            try:
                tpu_data = tpunicorn.tpu.get_tpus(zone=zone, project=project)
                if tpu_data:
                    break

            except:
                continue
        
        selected_tpu = None
        tpu_name = os.environ.get('TPU_NAME', None)
        if not tpu_data:
            print('Failed to find a TPU - Ensure you have the correct GOOGLE_APPLICATION_CREDENTIALS set for your project')
            sys.exit()
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