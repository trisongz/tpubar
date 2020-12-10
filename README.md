# TPUBar

 Google Cloud TPU Utilization Bar for Training Models
 
<p align="center">
    <br>
    <img src="https://github.com/trisongz/tpubar/raw/master/docs/tpubar_img.png"/>
    <br>
<p>


```shell
# from pypi
pip install --upgrade tpubar

# from src
pip install --upgrade git+https://github.com/trisongz/tpubar.git
```

## Quickstart

```python3
!pip install --upgrade tpubar
!pip install --upgrade git+https://github.com/trisongz/tpubar.git

# Option #1 on Colab

!tpubar test # you will be prompted to authenticate with GCE on Colab

# Option #2 on Colab

from google.colab import auth
from tpubar import TPUMonitor
import os

auth.authenticate_user()

monitor = TPUMonitor(tpu_name=os.environ.get('TPU_NAME', None), profiler='v2')

# your training code below

monitor.start()

for x in dataset:
    ops(x)
    print(monitor.current_stats)

# Option #3 in Terminal/CLI - (Non Colab/Remote VM/Your Desktop)
tpubar test tpu-name

```

## API Quickstart

```python3
from tpubar import TPUMonitor

'''
default args
- tpu_name = None, (str) name of a TPU you want to query, in case of multiple active TPUs
- project = None, (str) gcp project name
- profiler = 'v1', (str) options are ['v1', 'v2']
    - v1: for Non-Colab, Pytorch, Tensorflow Estimator (TF1), and Non-Tensorflow TPU Queries
    - v2: Colab, Tensorflow 2+
- refresh_secs = 10, (int) how many seconds between each query
- fileout = None, (str) path where tqdm goes to, defaults to sys.stdout
- verbose = False, (bool) prints current_stats every query if True
- disable = False, (bool) disables TPU Bars if True, useful if only stats want to be captured

# Colors can be defined using standard cli colors or hex (e.g. 'green' or ' #00 ff00')
- tpu_util = 'green', (str) color for TPU MXU Bar
- tpu_secondary = 'yellow', (str) color for second TPU Bar [Memory for v1, Active Time for v2]
- cpu_util = 'blue', (str) color for CPU Utilization Bar
- ram_util = 'blue' (str) color for RAM Utilization Bar

'''
monitor = TPUMonitor(tpu_name=None, project=None, profiler='v1', refresh_secs=10, fileout=None, verbose=False, disable=False, tpu_util='green', tpu_secondary='yellow', cpu_util='blue', ram_util='blue')

monitor.start()

# Can be called to retrieve stats, use stats.get(var, '') to avoid errors since Idle Time and Idle String don't return anything until after full TPU initialization.
'''
# Stats available

- v1 returns {'tpu_mxu': float, 'tpu_mem_per': float 'tpu_mem_used': float, 'tpu_mem_str': str, 'cpu_util': float, 'ram_util': float, 'ram_util_str': str}
- v2 returns {'tpu_mxu': float, tpu_mxu_str': str, 'tpu_idle_time': float, 'tpu_idle_str': str, 'cpu_util': float, 'ram_util': float, 'ram_util_str': str}
# Example
'v1': {'tpu_mxu': 52.88895420451343, 'tpu_mem_per': 100.0, 'tpu_mem_used': 198.5, 'tpu_mem_str': '198.50GB/127.96GB', 'cpu_util': 0.9, 'ram_util': 54.5, 'ram_util_str': '49.43GB/96.00GB'}

'''
stats = monitor.current_stats
tpu_mxu = stats.get('tpu_mxu', '')


```

## CLI Quickstart

The commands can be run remotely or on the same VM

```shell

# Monitor the TPU until Exit (cmd+c)
tpubar monitor [tpuname] --project [gcp_project] (optional)

# Test Run for 60 secs
tpubar test [tpuname] --project [gcp_project] (optional)

# Create or use an application key found in tpubar/auth.json
tpubar auth [adc_name] -l (list auths)

# Create new tmux session
tpubar sess [session_name]

# Attach your current window to the tmux session
tpubar attach [session_name]

# kill a tmux session
tpubar killsess [session_name]

```

<p align="center">
    <br>
    <img src="https://github.com/trisongz/tpubar/raw/master/docs/tpumonitor.png"/>
    <br>
<p>

## Notes

The reason for the 2 versions of TPUBar, 'v1' and 'v2' is due to how they use different API calls to get TPU metrics. Within Colab, only 'v2' works if you do not have TPUs in your Google Cloud project. Otherwise,  to avoid compatability issues.

- 'v1': is meant for TPU Projects running on GCE and/or Using Tensorflow < 2. Additionally, v1 can be called on a remote system (like your PC) to query your TPU running on GCE without being directly connected. Not yet tested, but should also be used in Pytorch training as well.

- 'v2' is meant for Colab and/or Tensorflow 2+, and uses tensorflow APIs, which require the system to be directly connected to the TPUs.

## Bonus

You can call 'tpubar sess new_session' in CLI to create a new tmux session and 'tpubar killsess new_session' to kill it.

## Contributors

[@shawwn](https://github.com/shawwn)

## Acknowledgements

[Tensorflow Research Cloud](https://www.tensorflow.org/tfrc) for providing TPU Resources