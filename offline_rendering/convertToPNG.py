#! /usr/bin/env python

import sys
import subprocess
import os

def convert():
	cwd = os.getcwd();
	files = [ f for f in os.listdir(cwd) if (f.endswith('exr')) ]
	for f in files:
		png = f.split('.')[0] + ".png"
		if not os.path.isfile(png):
			cmds = ['exrtopng', os.path.join(cwd,f), os.path.join(cwd,png)];
			print(cmds)
			subprocess.call(cmds)


if __name__ == "__main__":
	convert()