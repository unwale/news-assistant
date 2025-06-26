import subprocess

import torch

SAMPLE_RATE = 16000


def convert_to_torch_tensor(file_obj, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "-",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    pcm_data, _ = process.communicate(input=file_obj.read())
    tensor = torch.frombuffer(pcm_data, dtype=torch.int16).float() / 32768.0
    return tensor
