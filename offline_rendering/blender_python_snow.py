#! /usr/bin/env python
# Integration plugin for loading snow simulation data into Blender
# and exporting it via Mitsuba.


bl_info = {
    "name": "My Script",
    "description": "Import or Export Snow Simulation",
    "author": "evjang, mliberma, taparson, wyegelwe",
    "version": (1, 0),
    "blender": (2, 69, 0),
    "location": "File > Import > Snow Simulation",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "wyegelwel.github.io/snow",
    "category": "Import-Export"}


import os
import time
import bpy

class SnowImporter(object):
	def __init__(self):
		pass


	def import_scenefile():
		"""
		imports an XML
		"""
		pass


	def add_collider():
		"""
		adds collider geom and insert keyframes.
		"""
		pass

	def add_particle_system():
		"""
		create box with snow interior.
		"""
		pass

class MitsubaExporter(object):
	"""
	Uses the mtsblend plugin to export 1 .xml file for each frame 
	of animation. Paths to VOL files are injected during this step.
	"""
	def __init__(self):
		pass





def menu_func_import(self, context):
    pass

def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
	print('hello world')
	register()