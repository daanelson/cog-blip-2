import subprocess
import tarfile
import os


def download_weights(url, dest):
    """ Download the tar file, extract the weights, and move them to the "dest" directory. """
    if not os.path.exists("/src/tmp.tar"):
        print("Downloading weights...")
        try:
            output = subprocess.check_output(["pget", url, "/src/tmp.tar"])
            print(output)
        except subprocess.CalledProcessError as e:
            # If download fails, clean up and re-raise exception
            print(e.output)
            raise e
    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(path="/src/tmp")
    tar.close()
    os.rename("/src/tmp/weights", dest)
    os.remove("/src/tmp.tar")
    os.rmdir("/src/tmp")