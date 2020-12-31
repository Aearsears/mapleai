# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 02:13:51 2020

"""
import ctypes
from ctypes import wintypes
import time

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
KEYEVENTF_UNICODE     = 0x0004
KEYEVENTF_SCANCODE    = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731
VK_TAB  = 0x09
VK_ESCAPE=0x1B
VK_MENU = 0x12
VK_RIGHT = 0x27
VK_LEFT= 0x25
VK_UP=0x26
VK_DOWN=0x28
VK_LCONTROL = 0xA2
VK_A = 0x41
VK_X = 0x58
VK_J = 0x4A
VK_S= 0x53
VK_ENTER= 0x0D
VK_G = 0x47

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx",          wintypes.LONG),
                ("dy",          wintypes.LONG),
                ("mouseData",   wintypes.DWORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk",         wintypes.WORD),
                ("wScan",       wintypes.WORD),
                ("dwFlags",     wintypes.DWORD),
                ("time",        wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg",    wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))
    _anonymous_ = ("_input",)
    _fields_ = (("type",   wintypes.DWORD),
                ("_input", _INPUT))

LPINPUT = ctypes.POINTER(INPUT)

def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args

user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT, # nInputs
                             LPINPUT,       # pInputs
                             ctypes.c_int)  # cbSize

# Functions

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

def AltTab():
    """Press Alt+Tab and hold Alt key for 2 seconds
    in order to see the overlay.
    """
    PressKey(VK_MENU)   # Alt
    PressKey(VK_TAB)    # Tab
    ReleaseKey(VK_TAB)  # Tab~
    time.sleep(2)
    ReleaseKey(VK_MENU) # Alt~

def enterA():
    time.sleep(5)
    PressKey(0x02)
    time.sleep(1)
    ReleaseKey(0x02)

def jump():
    PressKey(VK_MENU)
    ReleaseKey(VK_MENU)

def testfunction(hexcode):
    PressKey(hexcode)
    time.sleep(0.5)
    ReleaseKey(hexcode)
    
def moveRight():
    PressKey(VK_RIGHT)
    time.sleep(0.5)
    ReleaseKey(VK_RIGHT)
    
def moveLeft():
    PressKey(VK_LEFT)
    time.sleep(0.5)
    ReleaseKey(VK_LEFT)

def buff():
    PressKey(VK_J)
    time.sleep(0.5)
    ReleaseKey(VK_J)

def jumpoffladder():
    PressKey(VK_LEFT)
    PressKey(VK_MENU)
    time.sleep(0.5)
    ReleaseKey(VK_MENU)
    ReleaseKey(VK_LEFT)

def cc():
    #move all the way left
    PressKey(VK_LEFT)
    time.sleep(20)
    ReleaseKey(VK_LEFT)
    #then attempt to cc
    PressKey(VK_ESCAPE)
    time.sleep(0.5)
    ReleaseKey(VK_ESCAPE)
    PressKey(VK_ENTER)
    time.sleep(0.5)
    ReleaseKey(VK_ENTER)
    PressKey(VK_LEFT)
    time.sleep(0.5)
    ReleaseKey(VK_LEFT)
    PressKey(VK_ENTER)
    time.sleep(0.5)
    ReleaseKey(VK_ENTER)
    # #press enter two more times to ensure all windows closed
    # PressKey(VK_ENTER)
    # time.sleep(0.5)
    # ReleaseKey(VK_ENTER)
    # PressKey(VK_ENTER)
    # time.sleep(0.5)
    # ReleaseKey(VK_ENTER)
    

def teleup():
    PressKey(VK_UP)
    PressKey(VK_S)
    time.sleep(0.5)
    ReleaseKey(VK_S)
    ReleaseKey(VK_UP)

def teleright():
    PressKey(VK_RIGHT)
    PressKey(VK_S)
    time.sleep(0.5)
    ReleaseKey(VK_S)
    ReleaseKey(VK_RIGHT)

def teleleft():
    PressKey(VK_LEFT)
    PressKey(VK_S)
    time.sleep(0.5)
    ReleaseKey(VK_S)
    ReleaseKey(VK_LEFT)

def teledown():
    PressKey(VK_DOWN)
    PressKey(VK_S)
    time.sleep(0.5)
    ReleaseKey(VK_S)
    ReleaseKey(VK_DOWN)

def enter():
    PressKey(VK_ENTER)
    time.sleep(0.5)
    ReleaseKey(VK_ENTER)
    PressKey(VK_ENTER)
    time.sleep(0.5)
    ReleaseKey(VK_ENTER)

def loot():
    for i in range(0,25):
        PressKey(VK_X)
        time.sleep(0.1)
        ReleaseKey(VK_X)

def pressg():
    PressKey(VK_G)
    time.sleep(0.1)
    ReleaseKey(VK_G)


def attackFiveTimes():
    for i in range(0,25):
        attack()

def attack():
    PressKey(VK_A)
    time.sleep(0.1)
    ReleaseKey(VK_A)

def useMPPot():
    PressKey(VK_X)
    time.sleep(0.2)
    ReleaseKey(VK_X)
   
