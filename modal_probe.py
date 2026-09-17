import modal

img = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
app = modal.App("clayfly-probe")


@app.function(image=img, timeout=600)
def hello():
    import os
    import platform
    return {"cpu": os.cpu_count(), "py": platform.python_version()}


@app.local_entrypoint()
def main():
    print(hello.remote())
