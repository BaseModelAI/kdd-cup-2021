import subprocess


def train_cleora(dim: int, iter_: int, columns: str, input_filename: str, working_dir: str):
    """
    Training Cleora. See more details: https://github.com/Synerise/cleora/
    """
    command = ['./cleora-v1.1.0-x86_64-unknown-linux-gnu',
               '--columns', columns,
               '--dimension', str(dim),
               '-n', str(iter_),
               '--input', input_filename,
               '--output-dir', working_dir]
    subprocess.run(command, check=True)