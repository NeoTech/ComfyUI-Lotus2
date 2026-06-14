import subprocess
import sys
import pkg_resources

def check_hf_version():
    try:
        version = pkg_resources.get_distribution("huggingface_hub").version
        if pkg_resources.parse_version(version) < pkg_resources.parse_version("1.18.0"):
            print("Updating huggingface_hub to 1.18.0+...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub>=1.18.0"])
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=1.18.0"])