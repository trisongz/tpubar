import click
import json
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import time

from tpubar.utils import run_command

@click.group()
@click.pass_context
def cli(ctx, **kws):
    ctx.obj = kws

@cli.command('monitor')
@click.argument('tpu_name', type=click.STRING, default=os.environ.get('TPU_NAME', None))
@click.option('--project', type=click.STRING, default=None)
@click.option('-v', '--verbose', is_flag=True)
def monitor_tpubar(tpu_name, project, verbose):
    tpu_name = tpu_name if tpu_name else os.environ.get('TPU_NAME', None)
    if not tpu_name:
        tpu_name = click.prompt('Please enter a TPU Name', type=click.STRING)
        if not tpu_name:
            raise ValueError('Valid TPU Name must be selected')
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None):
        adc = click.prompt('Please enter a path to GOOGLE_APPLICATION_CREDENTIALS', type=click.STRING)
        if adc:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = adc
    
    click.echo(f'Monitoring TPU: {tpu_name} until cancelled.')
    from tpubar import TPUMonitor, env
    if env['colab']:
        monitor = TPUMonitor(tpu_name=tpu_name, project=project, profiler='v2', refresh_secs=3, verbose=verbose)
    else:
        monitor = TPUMonitor(tpu_name=tpu_name, project=project, profiler='v1', refresh_secs=3, verbose=verbose)

    monitor.start()
    while True:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            click.echo(f'\nShutting Down Monitor')
            monitor.close()
            sys.exit()

@cli.command('test')
@click.argument('tpu_name', type=click.STRING, default=os.environ.get('TPU_NAME', None))
@click.option('--project', type=click.STRING, default=None)
def test_tpubar(tpu_name, project):
    tpu_name = tpu_name if tpu_name else os.environ.get('TPU_NAME', None)
    if not tpu_name:
        tpu_name = click.prompt('Please enter a TPU Name', type=click.STRING)
        if not tpu_name:
            raise ValueError('Valid TPU Name must be selected')
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', None):
        adc = click.prompt('Please enter a path to GOOGLE_APPLICATION_CREDENTIALS', type=click.STRING)
        if adc:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = adc
    
    click.echo(f'Running Test for TPUBar on TPU {tpu_name}')
    from tpubar import TPUMonitor, env
    if env['colab']:
        monitor = TPUMonitor(tpu_name=tpu_name, project=project, profiler='v2', refresh_secs=3, verbose=True)
    else:
        #click.echo(f'{project}')
        monitor = TPUMonitor(tpu_name=tpu_name, project=project, profiler='v1', refresh_secs=3, verbose=True)

    monitor.start()
    for x in range(6):
        time.sleep(10)
    click.echo(f'\nCompleted Testing')

@cli.command('sess')
@click.argument('session_name', default='train')
def create_sess(session_name):
    _conda_exe = os.getenv('CONDA_EXE').replace('bin/conda', 'etc/profile.d/conda.sh')
    _conda_env = os.getenv('CONDA_DEFAULT_ENV', None)
    command = f'tmux new -d -s {session_name}'
    os.system(command)
    if _conda_env:
        command = f'tmux send-keys -t {session_name}.0 "source {_conda_exe} && conda deactivate && conda activate {_conda_env} && clear && cd {os.getcwd()}" ENTER'
        os.system(command)
    os.system(f'tmux a -t {session_name}')

@cli.command('attach')
@click.argument('session_name', default='train')
def attach_sess(session_name):
    command = f'tmux a -t {session_name}'
    os.system(command)

@cli.command('killsess')
@click.argument('session_name', default='train')
def kill_sess(session_name):
    click.echo(f'Killing {session_name}')
    command = f'tmux kill-session -t {session_name}'
    os.system(command)


def main(*args, prog_name='tpubar', auto_envvar_prefix='TPUBAR', **kws):
    cli.main(*args, prog_name=prog_name, auto_envvar_prefix=auto_envvar_prefix, **kws)

if __name__ == "__main__":
    main()