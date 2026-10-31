import ctypes
import ctypes.wintypes as wt

class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]

class DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wt.UINT), ("DeviceId", wt.UINT),
                ("SubSysId", wt.UINT), ("Revision", wt.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID)]

class DISPLAY_DEVICE(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("DeviceName", ctypes.c_wchar * 32), ("DeviceString", ctypes.c_wchar * 128),
                ("StateFlags", wt.DWORD), ("DeviceID", ctypes.c_wchar * 128), ("DeviceKey", ctypes.c_wchar * 128)]

class OUTPUT(ctypes.Structure):
    _fields_ = [("Adapter", DXGI_ADAPTER_DESC), ("NumOutputs", wt.UINT), ("Description", ctypes.c_wchar * 128),
                ("AttachedToDesktop", wt.INT), ("Rotation", wt.INT), ("DesktopImage", wt.HANDLE)]

gdi = ctypes.WinDLL("user32.dll")
gdi.EnumDisplayDevicesW.restype = wt.BOOL
dev = DISPLAY_DEVICE()
dev.cb = ctypes.sizeof(dev)
for i in range(8):
    if not gdi.EnumDisplayDevicesW(None, i, ctypes.byref(dev), 0):
        break
    out = OUTPUT()
    ok = gdi.EnumDisplayDevicesW(dev.DeviceName, 0, ctypes.byref(out), 0)
    name = out.Adapter.Description or dev.DeviceString
    print("%s -> %s dedicated=%d MB shared=%d MB luid=%08x:%08x ok=%d" % (
        dev.DeviceName, name, out.Adapter.DedicatedVideoMemory // 1048576,
        out.Adapter.SharedSystemMemory // 1048576,
        out.Adapter.AdapterLuid.HighPart & 0xffffffff, out.Adapter.AdapterLuid.LowPart, ok))
