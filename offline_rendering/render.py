#!/usr/bin/env python

import argparse
import subprocess

def printHelp():
	print('''
Usage: 
divides *.xml into n jobs, and renders the b'th block.
For example, 

python render.py -b 3 -n 10 teapot_0000.xml, teapot_0001.xml, ... teapot_0099.xml

renders out teapot_0020.xml ... teapot_0029.xml 

to render out all the frames, just do python render teapot_*.xml
''')

def main():
	parser = argparse.ArgumentParser(description='Parallel Render Mitsuba Frames to movie')
	parser.add_argument('-i',type=int, help="bth block to render")
	parser.add_argument('-b',type=int, help="number of blocks to render")
	parser.add_argument('xmlFiles', type=str, nargs='+')
	
	try:
		args = parser.parse_args()
	except:
		printHelp()

	i = args.i
	b = args.b
	xmlFiles = args.xmlFiles
	n = len(xmlFiles)
	if i is not None and b is not None:
		print("i=",i)
		print("b=",b)
		start = (i-1)*b
		end = (i)*b if (i != b) else n # any rounding errors are assigned to the last block.
		xmlFiles = xmlFiles[start:end]
		print(xmlFiles)

	subprocess.call(['mitsuba','-xj'].extend(xmlFiles))
	
if __name__ == "__main__":
	main()