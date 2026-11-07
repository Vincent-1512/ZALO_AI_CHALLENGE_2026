"""Small Colab helpers: persistent logs and an actual CUDA/4-bit readiness check."""
from collections import deque
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def run_logged(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=40)
    with log_path.open('w', encoding='utf-8') as log:
        log.write('Command: ' + repr([str(arg) for arg in command]) + '\n')
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
    if code:
        raise RuntimeError(f'Exit code {code}. Full log: {log_path}\n' + ''.join(tail))
    return log_path


def check_runtime():
    import torch
    print('Python:', sys.version, flush=True)
    versions = {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'numpy')}
    print(json.dumps(versions, indent=2), flush=True)
    print('PyTorch CUDA:', torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU in Colab runtime settings.')
    # Import the actual training entry points, not just package metadata.
    from transformers import Trainer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import get_peft_model
    import bitsandbytes as bnb
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Compute capability:', torch.cuda.get_device_capability(0), flush=True)
    check_cuda_kernels(torch, bnb)
    print('Trainer imports and CUDA NF4 forward/backward/optimizer check passed.', flush=True)


def check_cuda_kernels(torch, bnb):
    layer = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.float16,
                             quant_type='nf4').to('cuda')
    inputs = torch.randn(2, 32, device='cuda', dtype=torch.float16, requires_grad=True)
    layer(inputs).float().square().mean().backward()
    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError('4-bit CUDA backward produced invalid gradients.')
    parameter = torch.nn.Parameter(torch.ones(4096, device='cuda'))
    optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-4)
    parameter.square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(parameter).all():
        raise RuntimeError('8-bit optimizer produced invalid parameters.')


def latest_complete_checkpoint(folder, require_scaler=False):
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                'adapter_config.json']
    if require_scaler:
        required.append('scaler.pt')
    candidates = [path for path in Path(folder).glob('checkpoint-*')
                  if path.is_dir() and path.name.split('-')[-1].isdigit()]
    for path in sorted(candidates, key=lambda p: int(p.name.split('-')[-1]), reverse=True):
        files_ok = all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required)
        weights_ok = any((path / name).is_file() and (path / name).stat().st_size > 0
                         for name in ('adapter_model.safetensors', 'adapter_model.bin'))
        try:
            state = json.loads((path / 'trainer_state.json').read_text())
            step_ok = state['global_step'] == int(path.name.split('-')[-1])
        except (OSError, ValueError, KeyError, TypeError):
            step_ok = False
        if files_ok and weights_ok and step_ok:
            return path
        print(f'Skipping incomplete checkpoint: {path}', flush=True)
    raise FileNotFoundError(f'No complete training checkpoint in {folder}')


if __name__ == '__main__':
    check_runtime()
