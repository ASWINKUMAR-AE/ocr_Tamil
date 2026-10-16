from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import shutil

weights = Path.home() / '.model_weights'
weights.mkdir(exist_ok=True)
base = 'https://huggingface.co/GnanaPrasath/ocr_tamil/resolve/main/'

def download(filename):
    destination = weights / filename
    if destination.exists():
        print(f'Already available: {filename}', flush=True)
        return
    print(f'Downloading {filename}', flush=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    with urlopen(base + filename, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)
    print(f'Ready: {filename} ({destination.stat().st_size:,} bytes)', flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download, ['parseq_tamil.pt', 'craft_mlt_25k.pth']))

shutil.copyfile(weights / 'parseq_tamil.pt', weights / 'parseq_tamil_v6.ckpt')
print('Tamil and CRAFT checkpoints ready.', flush=True)
