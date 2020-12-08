# tpubar
 Google Cloud TPU Utilization Bar for Training Models

I wouldn't really use this unless you know what you're doing

```shell
pip install --upgrade git+https://github.com/trisongz/tpubar.git
```


```python3
from tpubar import TPUMonitor

# if fileout is None, uses sys.stdout
# colors can be defined using standard cli colors or hex (e.g. 'green' or ' #00 ff00')

monitor = TPUMonitor(refresh_secs=10, fileout=None, verbose=True, tpu_util='green', tpu_secondary='yellow', cpu_util='blue', ram_util='blue')
monitor.start()

# Can be called to retrieve stats
stats = monitor.current_stats

# Use stats.get(var, '') to avoid errors since Idle Time and Idle String don't return anything until after full TPU initialization.

returns {
    'idle_time': float, # (TF2 or Colab)
    'idle_string', str, # (TF2 or Colab)
    'mxu': float,
    'mxu_string': str # (TF2 or Colab)
    'tpu_mem': float? # (TF 1)
}

```

## Contributors

[@shawwn](https://github.com/shawwn)

## Acknowledgements

[Tensorflow Research Cloud](https://www.tensorflow.org/tfrc) for providing TPU Resources