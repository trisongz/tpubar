import psutil
import platform

from tpubar.utils import run_command

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