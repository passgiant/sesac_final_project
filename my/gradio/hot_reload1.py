import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script_name):
        self.script_name = script_name
        self.process = None
        self.start_app()

    def start_app(self):
        if self.process:
            self.process.terminate()
        self.process = subprocess.Popen([sys.executable, self.script_name])

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"{event.src_path} changed, reloading...")
            self.start_app()

if __name__ == "__main__":
    script_name = "test14.py"  # Gradio 스크립트 이름
    event_handler = ReloadHandler(script_name)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(script_name)), recursive=False)
    observer.start()
    print("Watching for changes...")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
