from __future__ import annotations

import bpy
from bpy.types import Panel, UIList

from ..utils import env
from ..core import qremesh


class HALLWAYAVATAR_UL_layers(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index=0):
        label = item.semantic_label or "unclassified"
        if item.skipped:
            layout.label(text=f"{item.layer_name} ({item.skip_reason})", icon="X")
        else:
            layout.label(text=f"{item.layer_name} -> {label}", icon="IMAGE_DATA")


class HALLWAYAVATAR_PT_main(Panel):
    bl_label = "Hallway Avatar Gen"
    bl_idname = "HALLWAYAVATAR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Hallway"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        state = context.scene.hallway_avatar_state
        backend_status = env.psd_backend_status()

        source_box = layout.box()
        source_box.label(text="See-through PSD Source")
        source_box.prop(state, "source_psd_path", text="")
        source_box.operator("hallway_avatar.import_psd", icon="FILE_IMAGE")

        backend_box = layout.box()
        backend_box.label(text=f"PSD Backend: {backend_status}")
        if backend_status != "ready":
            backend_box.label(text="Place local parser and silhouette tracing wheels in the extension folder.")
            backend_box.operator("hallway_avatar.install_psd_backend", icon="IMPORT")

        options = layout.box()
        options.label(text="Import Options")
        options.prop(state, "ignore_hidden_layers")
        options.prop(state, "ignore_empty_layers")
        options.prop(state, "keep_tiny_named_parts")
        options.prop(state, "min_visible_pixels")
        options.prop(state, "mesh_grid_resolution")
        options.prop(state, "replace_existing")
        options.prop(state, "auto_bind_on_build")

        remesh_box = layout.box()
        remesh_box.label(text=f"Quad Remesh: {qremesh.runtime_status()}")
        remesh_box.prop(state.qremesh_settings, "auto_on_import")
        remesh_box.prop(state.qremesh_settings, "target_quad_count")
        remesh_box.prop(state.qremesh_settings, "unsubdivide_iterations", slider=True)
        remesh_box.prop(state.qremesh_settings, "unsubdivide_target_count")
        remesh_box.prop(state.qremesh_settings, "target_count_as_input_percentage")
        remesh_box.prop(state.qremesh_settings, "target_edge_length")
        remesh_box.prop(state.qremesh_settings, "adaptive_size")
        remesh_box.prop(state.qremesh_settings, "adapt_quad_count")
        remesh_box.label(text="By default, only front/back hair layers are quad remeshed.")
        remesh_box.label(text="Target Edge Length is converted into an estimated quad count for qmesh.")
        loop_box = remesh_box.box()
        loop_box.label(text="Edge Loops Control")
        loop_box.prop(state.qremesh_settings, "use_vertex_color_map")
        loop_box.prop(state.qremesh_settings, "use_materials")
        loop_box.prop(state.qremesh_settings, "use_normals_splitting")
        loop_box.prop(state.qremesh_settings, "autodetect_hard_edges")
        misc_box = remesh_box.box()
        misc_box.label(text="Misc")
        symmetry = misc_box.row(align=True)
        symmetry.label(text="Symmetry")
        symmetry.prop(state.qremesh_settings, "symmetry_x", text="X", toggle=True)
        symmetry.prop(state.qremesh_settings, "symmetry_y", text="Y", toggle=True)
        symmetry.prop(state.qremesh_settings, "symmetry_z", text="Z", toggle=True)
        advanced_box = remesh_box.box()
        advanced_box.prop(state.qremesh_settings, "show_advanced_filters", text="Advanced Remesh Filters")
        if state.qremesh_settings.show_advanced_filters:
            advanced_box.label(text="See-through categories to quad remesh")
            hair_row = advanced_box.row(align=True)
            hair_row.prop(state.qremesh_settings, "remesh_front_hair", toggle=True)
            hair_row.prop(state.qremesh_settings, "remesh_back_hair", toggle=True)
            body_row = advanced_box.row(align=True)
            body_row.prop(state.qremesh_settings, "remesh_topwear", toggle=True)
            body_row.prop(state.qremesh_settings, "remesh_handwear", toggle=True)
            body_row.prop(state.qremesh_settings, "remesh_bottomwear", toggle=True)
            body_row.prop(state.qremesh_settings, "remesh_legwear", toggle=True)
            body_row.prop(state.qremesh_settings, "remesh_footwear", toggle=True)
            extras_row = advanced_box.row(align=True)
            extras_row.prop(state.qremesh_settings, "remesh_face_head", toggle=True)
            extras_row.prop(state.qremesh_settings, "remesh_tail", toggle=True)
            extras_row.prop(state.qremesh_settings, "remesh_wings", toggle=True)
            extras_row.prop(state.qremesh_settings, "remesh_objects", toggle=True)
            advanced_box.prop(state.qremesh_settings, "remesh_unclassified", toggle=True)
        remesh_box.label(text="Hallway now uses the vendored remesher runtime inside this extension.")
        remesh_box.operator("hallway_avatar.remesh_imports", icon="MOD_REMESH")

        rigging = layout.box()
        rigging.label(text="Rigging")
        rigging.operator("hallway_avatar.build_armature", icon="ARMATURE_DATA")
        rigging.operator("hallway_avatar.bind_weights", icon="MOD_ARMATURE")
        rigging.operator("hallway_avatar.run_pipeline", icon="PLAY")

        roadmap = layout.box()
        roadmap.label(text="Roadmap")
        roadmap.label(text="Stretchy-style PSD rigging is available as a first pass.")
        roadmap.label(text="Full 2.5-D avatar generation via See-through is still coming later.")

        summary = layout.box()
        summary.label(text="Summary")
        summary.label(text=f"Imported: {state.imported_count}")
        summary.label(text=f"Remeshed: {state.remeshed_count}")
        summary.label(text=f"Skipped: {state.skipped_count}")
        summary.label(text=f"Classified: {state.classified_count}")
        if state.last_report:
            summary.label(text=state.last_report)

        if state.layer_items:
            layout.template_list(
                "HALLWAYAVATAR_UL_layers",
                "",
                state,
                "layer_items",
                state,
                "active_layer_index",
                rows=8,
            )


classes = (
    HALLWAYAVATAR_UL_layers,
    HALLWAYAVATAR_PT_main,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
