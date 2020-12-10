import os
env = dict()

try:
    import google.colab
    env['colab'] = True
except ImportError:
    env['colab'] = False

import tensorflow as tf
env['tf2'] = True if tf.__version__.startswith('2') else False
try:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.python.framework import errors
    env['profiler'] = True
except ImportError:
    env['profiler'] = False

if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    if env['colab']:
        print('Authenticating with Google Cloud Engine to access TPUs')
        from google.colab import auth
        auth.authenticate_user()
    else:
        print('No GOOGLE_APPLICATION_CREDENTIALS Detected as Environment Variable. You may run into TPU Authentication issues.')

env['dir'] = os.path.abspath(os.path.dirname(__file__))
import tpubar.utils
import tpubar.host
import tpubar.network
import tpubar.monitor
from tpubar.monitor import TPUMonitor