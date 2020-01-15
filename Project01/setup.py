"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""
from setuptools import setup

APP = ['gui.py']
DATA_FILES = ['common.py', 'matches1.py', 'matches2.py', ('',['resources'])]
OPTIONS = {'includes': ['collections','re','random','sys','sip','PyQt5.QtGui','PyQt5.QtWidgets'],}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)