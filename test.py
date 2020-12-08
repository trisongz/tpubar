from tpubar import TPUMonitor
import time

monitor = TPUMonitor(tpu_name='t5-xlarge', profiler='v1', verbose=True)
monitor.start()

x = 1
while True:
    time.sleep(5)
    x += 1
    #if x % 3 == 0:
    #    print('trying to mess up stuff yea')
