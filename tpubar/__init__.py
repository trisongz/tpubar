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
        import google.auth
        creds, project_id = google.auth.default()
        if creds:
            default_adc = os.path.join(os.environ.get('HOME', env['dir']), 'adc.json')
            creds.expiry = None
            creds = dict(creds.__dict__)
            _creds = dict()
            for k in creds.keys():
                if k.startswith('_'):
                    _creds[k[1:]] = creds[k]
                else:
                    _creds[k] = creds[k]

            _creds['type'] = 'authorized_user' if _creds.get('refresh_token', None) else 'service_account'
            json.dump(_creds, open(default_adc, 'w'))
            auths['DEFAULT_ADC'] = default_adc
            print(f'Found ADC Credentials Implicitly. Saving to {default_adc} for future runs.\nSet GOOGLE_APPLICATION_CREDENTIALS={default_adc} in Environment to allow libraries like Tensorflow to locate your ADC.')
            update_auth(auths)

        else:
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