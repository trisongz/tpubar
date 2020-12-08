# tpubar
 Google Cloud TPU Utilization Bar for Training Models

I wouldn't really use this unless you know what you're doing

```shell
pip install --upgrade git+https://github.com/trisongz/tpubar.git
```


```python3
from tpubar import TPUMonitor

monitor = TPUMonitor()
monitor.start()

# Can be called to retrieve stats
stats = monitor.current_stats

Use stats.get(var, '') to avoid errors since Idle Time and Idle String don't return anything until after full TPU initialization.

returns {
    'idle_time': float,
    'idle_string', str,
    'mxu': float,
    'mxu_string': str
}

```
