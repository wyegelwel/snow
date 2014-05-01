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
import sys
from xml.dom import minidom

sys.path.append(os.path.dirname('~/.config/blender/2.69/scripts/addons/mtsblend'))
from mtsblend.export.scene import SceneExporterProperties, SceneExporter


class SnowImporter(object):
	"""
	Parses our custom sceneFile format and adds a 
	proxy for the ParticleSystem into the scene.
	"""
	def __init__(self):
		pass

	def import_scenefile():
		"""
		imports an XML
		"""
# TODO - set the start and end frame of the blender file.
# scn = bpy.context.scene
# scn.frame_start = 1
# scn.frame_end = 101
		pass

	def add_collider():
		"""
		adds collider geom and insert keyframes.
		"""
		pass
# linear animation interpolation!!!
# keyInterp = context.user_preferences.edit.keyframe_new_interpolation_type
# context.user_preferences.edit.keyframe_new_interpolation_type ='LINEAR'

# temp_ob.keyframe_insert(data_path='location', frame=(cf))
# temp_ob.keyframe_insert(data_path='rotation_euler', frame=(cf))

# context.user_preferences.edit.keyframe_new_interpolation_type = keyInterp

	def add_particle_system():
		"""
		create box with snow interior.
		"""
		# bpy.ops.object.empty_add(type='CUBE', view_align=False, location=())
		# designate it with special label
		pass

class MitsubaSnowExporter(object):
	"""
	Simply makes a call to the mtsblend plugin to export 1 .xml file for each frame 
	of animation. Paths to VOL files are injected during this step.
	"""
	def __init__(self):
		pass

	def export(self):
		# bpy.ops.anim.change_frame(frame = 17)
		# bpy.ops.anim.keyframe_insert_menu(type='Translation')

		# As of 30 April 2014, mtsblend's "bpy.ops.export.mitsuba" operator is broken.
		# however rendering obviously works by writing out the correct XMLs, so we will
		# mimic that part of the pipeline.
		scene = bpy.context.scene
		scene_exporter = SceneExporter()
		scene_exporter.properties.directory = self.output_dir
		scene_exporter.properties.filename = self.output_filename # dont add the xml extension, it gets added later.
		scene_exporter.properties.api_type = 'FILE'			# Set export target
		scene_exporter.properties.write_files = True		# Use file write decision from above
		scene_exporter.properties.write_all_files = False		# Use UI file write settings
		scene_exporter.set_scene(scene)
		export_result = scene_exporter.export()

################################################################################################

class SnowImportOperator(bpy.types.Operator):
	""" Load a Snow Simulation Scene """	# tooltip
	bl_idname = "snow.import"				# unique identifier for buttons and menu items to reference
	bl_label = "Import Snow Simulation"		# display name in interface
	bl_options = {'REGISTER', 'UNDO'}		# enable undo
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'

	filepath = bpy.props.StringProperty(subtype='FILE_PATH', options={'HIDDEN',})

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	# entrypoint
	def execute(self, context):
		scene = bpy.context.scene
		# do stuff here!!!!
		# for obj in scene.objects:
		# 	obj.location.x += 1.0
		print(self.filepath)
		context.scene.update()
		return {'FINISHED'}	# this lets blender know the operator finished successfully.

class SnowExportOperator(bpy.types.Operator):
	""" Export Mitsuba Snow """ 
	bl_idname = "snow.export"				# unique identifier for buttons and menu items to reference
	bl_label = "Export Mitsuba Snow"		# display name in interface
	bl_options = {'REGISTER', 'UNDO'}		# enable undo
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'

	filepath = bpy.props.StringProperty(subtype='FILE_PATH', options={'HIDDEN',})

	def invoke(self, context, event):
		context.window_manager.fileselect_add(self)
		return {'RUNNING_MODAL'}

	def execute(self, context):
		# scene = bpy.context.scene
		print(self.filepath)
		context.scene.update()
		return {'FINISHED'}

def menu_import_func(self, context):
    self.layout.operator_context = 'INVOKE_DEFAULT'
    self.layout.operator(SnowImportOperator.bl_idname, text=SnowImportOperator.bl_label)

def menu_export_func(self, context):
	self.layout.operator_context = 'INVOKE_DEFAULT'
	self.layout.operator(SnowExportOperator.bl_idname, text=SnowExportOperator.bl_label)

def register():
	print('Registering Plugin')
	# importing
	bpy.utils.register_class(SnowImportOperator)
	bpy.types.INFO_MT_file_import.append(menu_import_func)
	# exporting
	bpy.utils.register_class(SnowExportOperator)
	bpy.types.INFO_MT_file_import.append(menu_export_func)

def unregister():
#	bpy.utils.unregister_module(__name__)
	print('Unregistering Plugin')
	bpy.utils.unregister_class(SnowImportOperator)
	bpy.utils.unregister_class(SnowExportOperator)


################################################################################################

# entry point of script
# allows this to be run from the Blender internal text editor without having to install
if __name__ == "__main__":
	print('hello world')
	register()

# run this to find addon path locations in python console
# import addon_utils
#print(addon_utils.paths())