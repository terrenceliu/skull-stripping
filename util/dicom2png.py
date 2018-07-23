import pydicom
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from multiprocessing import Process, cpu_count, Value, Lock
import sys


# Atomic Counter
class Counter(object):
    def __init__(self, total, initval=0, interval=500):
        self.val = Value('i', initval)
        self.lock = Lock()
        self.total = total
        self.interval = interval

    def increment(self):
        with self.lock:
            self.val.value += 1

            # Logging
            if (self.val.value % self.interval == 0):
                print("[Counter] Finished %f%%. index = %d" % (1.0 * self.val.value / self.total * 100, self.val.value))

    def value(self):
        with self.lock:

            return self.val.value


# Global variables
input_path = "..\\data\\dicom\\1021018"
output_path = "..\\data\\dicom\\png"



def prepare_workload(input_path, chunks):
    """

    :param input_path:
    :param chunks:
    :return:
    """
    data = []
    for root, dirs, files in os.walk(input_path):
        for f in files:
            # Only extracts dicom files
            if f.endswith(".dcm"):
                data.append(os.path.join(root, f))

    workload = []
    chunk_size = int(len(data) / chunks)
    for i in range(chunks):
        if (i == 0):
            workload.append(data[:chunk_size])
        elif (i == chunks - 1):
            workload.append(data[i * chunk_size:])
        else:
            workload.append(data[i * chunk_size: (i + 1) * chunk_size])
    return workload



def run(workload, counter):
    global input_path
    global output_path

    for f in workload:
        # Increment counter
        counter.increment()

        # Output name
        output_name = output_path + "\\" + f.split("\\")[-1].rstrip(".dcm") + ".png"

        # Read dicom
        ds = pydicom.dcmread(f)

        try:
            pixel = ds.pixel_array
        except Exception:
            print(sys.exc_info()[0])
            continue

        max_val = np.max(pixel)
        min_val = np.min(pixel)

        # Normalize
        pixel = (pixel - min_val) / max_val * 255

        img = Image.fromarray(pixel)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(output_name)





def temp():
    count = 0
    for f in input_path:

        count += 1

        if (count % 500 == 0):
            print("[dcm2png] Finished %d. %f%%" % (count, 1.0 * count / files_count * 100))

        # Output name
        output_name = output_path + "\\" + f.split("\\")[-1].rstrip(".dcm") + ".png"

        ds = pydicom.dcmread(f)
        pixel = ds.pixel_array

        max_val = np.max(pixel)
        min_val = np.min(pixel)

        # Normalize
        pixel = (pixel - min_val) / max_val * 255

        img = Image.fromarray(pixel)

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(output_name)


if __name__ == "__main__":
    workload = prepare_workload(input_path, cpu_count())

    files_count = 0
    for i in workload:
        files_count += len(i)

    print("Workload(%d): %d" % (len(workload), files_count))
    print("CPU: %d" % cpu_count())

    counter = Counter(total=files_count, interval=100)

    pool = []
    for work in workload:
        p = Process(target=run, args=(work, counter))
        p.start()
        pool.append(p)


    for p in pool:
        p.join()

    print("Finish. Counter: ", counter.value())








