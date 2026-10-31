import ctypes
import subprocess
import time

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

cmd = ["build/sdvkgen.exe", "-m", "E:/ofc-data/noobai", "--spv", "build/shaders", "--param-mb", "2944",
       "--vpred", "--spacing", "trailing", "--steps", "1", "--cfg", "3.0", "--width", "256", "--height", "256",
       "--out", "output/probe.png", "--no-vae", "--no-latent", "--prompt", "x"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, p.pid)
peak = 0
peakws = 0
t0 = time.time()
while p.poll() is None:
    m = PMC()
    m.cb = ctypes.sizeof(m)
    ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(m), m.cb)
    peak = max(peak, m.PagefileUsage)
    peakws = max(peakws, m.WorkingSetSize)
    if time.time() - t0 > 180:
        p.kill()
        break
    time.sleep(0.02)
out = p.stdout.read()
print(out[-1200:])
print("peak commit %.0f MB, peak working set %.0f MB" % (peak / 1048576, peakws / 1048576))
