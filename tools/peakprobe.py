import ctypes
import subprocess
import sys
import threading
import time


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


cmd = [r"build\sdvkgen.exe", "--textcheck", "-m", "E:/ofc-data/noobai", "--spv", "build/shaders",
       "--prompt", "hello"]
if len(sys.argv) > 1:
    cmd = sys.argv[1:]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, p.pid)
peak = [0, 0]
t0 = time.time()


def mon():
    while p.poll() is None:
        m = PMC()
        m.cb = ctypes.sizeof(m)
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(m), m.cb)
        peak[0] = max(peak[0], m.PagefileUsage)
        peak[1] = max(peak[1], m.WorkingSetSize)
        time.sleep(0.01)


th = threading.Thread(target=mon)
th.start()
out = p.communicate()[0].decode(errors="replace")
th.join()
print(out[-2500:])
print("peak commit %.0f MB  peak working set %.0f MB  elapsed %.1f s" % (peak[0] / 1048576, peak[1] / 1048576, time.time() - t0))
