from tpubar import TPUMonitor
import time

monitor = TPUMonitor(tpu_name='t5-xlarge', profiler='v1', verbose=False, disable=True)
monitor.start()

x = 1
while True:
    time.sleep(5)
    x += 1
    print(monitor.current_stats)
