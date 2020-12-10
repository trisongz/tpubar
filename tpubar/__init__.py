import os
import json
env = dict()

try:
    import google.colab
    env['colab'] = True
except ImportError:
    env['colab'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
env['tf2'] = True if tf.__version__.startswith('2') else False
try:
    from tensorflow.python.profiler import profiler_client
    from tensorflow.python.framework import errors
    env['profiler'] = True
except ImportError:
    env['profiler'] = False

env['dir'] = os.path.abspath(os.path.dirname(__file__))
env['auth_path'] = os.path.join(env['dir'], 'auth.json')
auths = json.load(open(env['auth_path']))

def update_auth(updated_auths):
    json.dump(updated_auths, open(env['auth_path'], 'w'), indent=1)

if auths.get('DEFAULT_ADC', None):
    if auths['DEFAULT_ADC'] != 'IMPLICIT':
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auths['DEFAULT_ADC']

elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None):
    if not auths.get('DEFAULT_ADC', None):
        auths['DEFAULT_ADC'] = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        update_auth(auths)
else:
    if env['colab']:
        print('Authenticating with Google Cloud Engine to access TPUs')
        from google.colab import auth
        auth.authenticate_user()
        auths['DEFAULT_ADC'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '/content/adc.json')
        update_auth(auths)

    else:
        try:
            import googleapiclient.discovery
            storage_client = googleapiclient.discovery.build('storage', 'v1')
            buckets = storage_client.buckets().list().execute()
            if buckets:
                auths['DEFAULT_ADC'] = 'IMPLICIT'
                update_auth(auths)

        except:
            print('No GOOGLE_APPLICATION_CREDENTIALS Detected as Environment Variable. Run "tpubar auth auth_name" to set your ADC. You may run into Issues otherwise.')

def set_auth(auth_name):
    if auth_name in auths.keys():
        print(f'Setting ADC to {auth_name}: {auths[auth_name]}')
        if auths[auth_name] in auths.values():
            auths['BACKUP_ADC_PATH'] = auths[auth_name]
        auths['DEFAULT_ADC'] = auths[auth_name]
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = auths[auth_name]
        update_auth(auths)
    else:
        print(f'Not able to find {auth_name} in Auth File. Update it first using "tpu auth {auth_name}".')


import tpubar.utils
import tpubar.host
import tpubar.network
import tpubar.monitor
from tpubar.monitor import TPUMonitor