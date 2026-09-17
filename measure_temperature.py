from datetime import datetime
import click
import serial
import threading
from pathlib import Path
import pandas as pd
import time
import re


running = True


class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name="keyboard-input-thread"):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input())

def callback(inp):
    global running
    if inp.strip().lower() == "q":
        running = False

kthread = KeyboardThread(callback)

@click.command()
@click.option("--comment", "-c", type=str, default="")
def main(comment):
    global running

    measurement_path = Path("measurements")
    measurement_path.mkdir(exist_ok=True, parents=True)
    filename = measurement_path / (f"temperature_{comment}_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")

    rows = {
        "time": [],
        "T_ambient": [],
        "T_object": []
    }
    swap_filename = filename.with_name("." + filename.name + ".swp")
    
    click.secho(f"Saving as {filename}, swapping as {swap_filename}")
    click.secho("Press q and hit 'enter' to quit", fg="yellow")

    with serial.Serial("COM1", 9600) as ser:
        ser.flush()
        ser.flushInput()
        ser.flushOutput()
        time.sleep(3)
        click.secho("Device connected, temperature acquisition running!", fg="green")
        while running:
            line = ser.readline()
            try:
                line = line.decode("utf-8").strip()
            except UnicodeDecodeError():
                click.secho(f"Malformed line: {line}, skipping", fg="red")
                continue
            pattern = r"^ambient=\d+\.\d+ object=\d+\.\d+$"
            if re.match(pattern, line):
                tokens = line.split(" ")
                T_ambient = float(tokens[0].split("=")[1])
                T_object = float(tokens[1].split("=")[1])
                rows["time"].append(datetime.now().strftime("%Y%m%d_%H%M%S"))
                rows["T_ambient"].append(T_ambient)
                rows["T_object"].append(T_object)
                df_swap = pd.DataFrame(rows)
                df_swap.to_csv(swap_filename)
            else:
                click.secho(f"Malformed line: {line}, skipping", fg="red")
        click.secho(f"Done, saving as {filename}")
        df_swap.to_csv(filename)
        swap_filename.unlink()


if __name__ == "__main__":
    main()
