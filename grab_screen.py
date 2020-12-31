# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:46:06 2020

"""
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
import win32gui,win32process


def set_window(process):
    
    hwin = win32gui.FindWindow(None, process)
    win32gui.SetWindowPos(hwin, win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOSIZE)
    win32gui.ShowWindow(hwin,5)
    #win32con.SWP_SHOWWINDOW | 
    #image = ImageGrab.grab(rect)
    # rect = win32gui.GetWindowPlacement(hwin)[-1]
    # newrect = (rect[0],rect[1]+30,rect[2]-16,rect[3]-29+30)
    # return newrect

def grab_screen(region=None,process=None):
    """
    returns numpy array (height,weight,colours)
    """
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    elif process:
        hwin = win32gui.FindWindow(None, process)
        rect = win32gui.GetWindowRect(hwin)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        width = w
        height = h
        left = x
        top = y
        if hwin == 0:
            print("The process doesn't exist, capturing the screen...")
            hwin = win32gui.GetDesktopWindow()
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def get_window_pid(title):
    hwnd = win32gui.FindWindow(None, title)
    threadid,pid = win32process.GetWindowThreadProcessId(hwnd)
    return pid