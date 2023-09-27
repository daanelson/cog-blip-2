import subprocess
import tarfile
import os

def download_weights(url, dest):
    if not os.path.exists("tmp.tar"):
        subprocess.check_call(["pget", url, "tmp.tar"])
    tar = tarfile.open("tmp.tar")
    tar.extractall(path="tmp")
    tar.close()
    # os.rename("tmp/weights", dest)
    # os.remove("tmp.tar")
    # os.rmdir("tmp")

if __name__ == "__main__":
    download_weights(
        url="https://storage.googleapis.com/replicate-weights/blip-2/weights.tar",
        dest="weights",
    )