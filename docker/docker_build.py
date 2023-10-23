#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "docker build -t lang-manip-training ."
    code = os.system(cmd)
