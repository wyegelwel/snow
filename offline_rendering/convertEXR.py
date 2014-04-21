#! /usr/bin/env python

# make sure /contrib/projects/exrtools/bin is in your path

import sys
import subprocess
import pdb

def exr2png(exr):
	fname = exr.split('.exr')[0]
	pngname = fname + '.png'
	cmd = ["exrtopng", exr, pngname];
	print(cmd)
	subprocess.call(cmd)

if __name__ == "__main__":
	exrs = sys.argv[1:]
	#pdb.set_trace()
	for e in exrs:
		exr2png(e)
	print("done")