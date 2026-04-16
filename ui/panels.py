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


def _draw_header(header: bpy.types.UILayout, text: str, *, icon: str, alert: bool = False) -> None:
    header.alert = alert
    header.label(text=text.upper(), icon=icon)


def _draw_reset_button(layout: bpy.types.UILayout, group: str) -> None:
    buttons = layout.row(align=True)
    buttons.scale_x = 0.43
    op = buttons.operator("hallway_avatar.reset_settings_group", text="Reset")
    op.group = group


def _draw_toggle_prop(
    layout: bpy.types.UILayout,
    data,
    data_path: str,
    prop_name: str,
    *,
    label: str | None = None,
) -> None:
    row = layout.row(align=True)
    row.use_property_split = False
    row.label(text=label or data.bl_rna.properties[prop_name].name)
    value = bool(getattr(data, prop_name))
    buttons = row.row(align=True)
    buttons.scale_x = 0.43
    on_op = buttons.operator("hallway_avatar.set_bool", text="ON", depress=value)
    on_op.data_path = data_path
    on_op.prop_name = prop_name
    on_op.value = True
    off_op = buttons.operator("hallway_avatar.set_bool", text="OFF", depress=not value)
    off_op.data_path = data_path
    off_op.prop_name = prop_name
    off_op.value = False


def _draw_group_title(layout: bpy.types.UILayout, text: str, *, icon: str = "BLANK1") -> None:
    row = layout.row()
    row.label(text=text, icon=icon)


def _draw_import_settings(layout: bpy.types.UILayout, state) -> None:
    state_path = "scene.hallway_avatar_state"
    import_top = layout.row(align=True)
    import_top.label(text="IMPORT OPTIONS", icon="SETTINGS")
    _draw_reset_button(import_top, "import_options")
    layout.separator()
    _draw_group_title(layout, "Layer Filtering", icon="IMAGE_DATA")
    _draw_toggle_prop(layout, state, state_path, "ignore_hidden_layers")
    _draw_toggle_prop(layout, state, state_path, "ignore_empty_layers")
    _draw_toggle_prop(layout, state, state_path, "keep_tiny_named_parts")
    layout.separator()
    _draw_group_title(layout, "Import Density", icon="MESH_DATA")
    layout.prop(state, "min_visible_pixels")
    layout.prop(state, "mesh_grid_resolution")
    layout.separator()
    _draw_group_title(layout, "Output Behavior", icon="OUTLINER_COLLECTION")
    _draw_toggle_prop(layout, state, state_path, "replace_existing")
    _draw_toggle_prop(layout, state, state_path, "auto_bind_on_build")
    layout.separator()

    alpha_header, alpha_panel = layout.panel_prop(state, "show_advanced_alpha_settings")
    _draw_header(alpha_header, "Advanced Import Alpha", icon="NODE_TEXTURE")
    if alpha_panel:
        threshold_header, threshold_panel = alpha_panel.panel_prop(state, "show_alpha_thresholds_section")
        _draw_header(threshold_header, "Transparency Controls", icon="TEXTURE")
        _draw_reset_button(threshold_header, "alpha_thresholds")
        if threshold_panel:
            _draw_group_title(threshold_panel, "Visible Pixel Thresholds", icon="IMAGE_DATA")
            threshold_panel.prop(state, "alpha_noise_floor")
            threshold_panel.prop(state, "visible_alpha_threshold")
            threshold_panel.separator()
            _draw_group_title(threshold_panel, "Adaptive Thresholding", icon="MODIFIER")
            _draw_toggle_prop(threshold_panel, state, state_path, "auto_alpha_threshold_boost")

        trace_header, trace_panel = alpha_panel.panel_prop(state, "show_trace_contrast_section")
        _draw_header(trace_header, "Trace Contrast Remap", icon="SHADING_RENDERED")
        _draw_reset_button(trace_header, "trace_contrast")
        if trace_panel:
            _draw_group_title(trace_panel, "Silhouette Trace Range", icon="CURVE_DATA")
            trace_panel.prop(state, "trace_contrast_low")
            trace_panel.prop(state, "trace_contrast_high")


def _draw_remesh_settings(layout: bpy.types.UILayout, remesh) -> None:
    remesh_path = "scene.hallway_avatar_state.qremesh_settings"
    _draw_group_title(layout, "Topology Target", icon="MOD_REMESH")
    layout.prop(remesh, "target_quad_count")
    layout.prop(remesh, "unsubdivide_iterations", slider=True)
    layout.prop(remesh, "unsubdivide_target_count")
    layout.separator()
    _draw_group_title(layout, "Sizing Strategy", icon="DRIVER_DISTANCE")
    layout.prop(remesh, "target_edge_length")
    layout.prop(remesh, "adaptive_size")
    _draw_toggle_prop(layout, remesh, remesh_path, "adapt_quad_count")
    _draw_toggle_prop(layout, remesh, remesh_path, "target_count_as_input_percentage")
    layout.separator()
    loops_top = layout.row(align=True)
    loops_top.label(text="Edge Loops Control", icon="MOD_EDGESPLIT")
    _draw_reset_button(loops_top, "edge_loops")
    layout.separator()
    _draw_toggle_prop(layout, remesh, remesh_path, "use_vertex_color_map")
    _draw_toggle_prop(layout, remesh, remesh_path, "use_materials")
    layout.separator()
    _draw_toggle_prop(layout, remesh, remesh_path, "use_normals_splitting")
    _draw_toggle_prop(layout, remesh, remesh_path, "autodetect_hard_edges")
    layout.separator()
    symmetry_top = layout.row(align=True)
    symmetry_top.label(text="Symmetry", icon="ORIENTATION_GLOBAL")
    _draw_reset_button(symmetry_top, "remesh_misc")
    layout.separator()
    _draw_toggle_prop(layout, remesh, remesh_path, "symmetry_x")
    _draw_toggle_prop(layout, remesh, remesh_path, "symmetry_y")
    _draw_toggle_prop(layout, remesh, remesh_path, "symmetry_z")


class HALLWAYAVATAR_PT_main(Panel):
    bl_label = "Hallway Avatar Gen"
    bl_idname = "HALLWAYAVATAR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Hallway"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        state = context.scene.hallway_avatar_state
        remesh = state.qremesh_settings
        backend_status = env.psd_backend_status()
        state_path = "scene.hallway_avatar_state"
        remesh_path = "scene.hallway_avatar_state.qremesh_settings"

        source_header, source_panel = layout.panel_prop(state, "show_source_section")
        _draw_header(source_header, "See-through PSD Source", icon="FILE_IMAGE", alert=True)
        if source_panel:
            source_row = source_panel.row(align=True)
            source_row.prop(state, "source_psd_path", text="")
            source_row.operator("hallway_avatar.select_psd_path", text="", icon="FILE_FOLDER")
            source_row.popover(
                panel="HALLWAYAVATAR_PT_import_popover",
                text="",
                icon="PREFERENCES",
            )

            import_button = source_panel.row()
            import_button.scale_y = 1.8
            import_button.operator("hallway_avatar.import_psd", icon="FILE_IMAGE")
            source_panel.separator()
            _draw_toggle_prop(source_panel, remesh, "scene.hallway_avatar_state.qremesh_settings", "auto_on_import")

        if backend_status != "ready":
            backend_header, backend_panel = layout.panel_prop(state, "show_backend_section")
            _draw_header(backend_header, "PSD Backend", icon="IMPORT", alert=True)
            if backend_panel:
                backend_panel.label(text=f"Status: {backend_status}")
                backend_panel.separator()
                backend_panel.label(text="Place local parser and silhouette tracing wheels in the extension folder.")
                backend_panel.operator("hallway_avatar.install_psd_backend", icon="IMPORT")

        remesh_header, remesh_panel = layout.panel_prop(remesh, "show_section")
        _draw_header(remesh_header, f"Quad Remesh: {qremesh.runtime_status()}", icon="MOD_REMESH", alert=True)
        if remesh_panel:
            main_top = remesh_panel.row(align=True)
            main_top.alert = False
            main_top.label(text="MAIN SETTINGS", icon="PREFERENCES")
            _draw_reset_button(main_top, "remesh_main")
            main_top.popover(panel="HALLWAYAVATAR_PT_remesh_popover", text="", icon="PREFERENCES")
            remesh_panel.separator()
            remesh_button = remesh_panel.row()
            remesh_button.scale_y = 1.8
            remesh_button.operator("hallway_avatar.remesh_imports", icon="MOD_REMESH")
            advanced_header, advanced_panel = remesh_panel.panel_prop(remesh, "show_advanced_filters")
        else:
            advanced_header, advanced_panel = layout.panel_prop(remesh, "show_advanced_filters")
        _draw_header(advanced_header, "Advanced Remesh Filters", icon="MODIFIER", alert=True)
        if advanced_panel:
            advanced_top = advanced_panel.row(align=True)
            advanced_top.label(text="See-through categories to quad remesh")
            _draw_reset_button(advanced_top, "remesh_filters")
            advanced_panel.separator()
            advanced_panel.label(text="By default, only front and back hair layers are remeshed.")
            advanced_panel.separator()
            hair_box = advanced_panel.box()
            _draw_group_title(hair_box, "Hair", icon="USER")
            _draw_toggle_prop(hair_box, remesh, remesh_path, "remesh_front_hair")
            _draw_toggle_prop(hair_box, remesh, remesh_path, "remesh_back_hair")
            advanced_panel.separator()
            wear_box = advanced_panel.box()
            _draw_group_title(wear_box, "Wearables", icon="OUTLINER_OB_ARMATURE")
            _draw_toggle_prop(wear_box, remesh, remesh_path, "remesh_topwear")
            _draw_toggle_prop(wear_box, remesh, remesh_path, "remesh_handwear")
            _draw_toggle_prop(wear_box, remesh, remesh_path, "remesh_bottomwear")
            _draw_toggle_prop(wear_box, remesh, remesh_path, "remesh_legwear")
            _draw_toggle_prop(wear_box, remesh, remesh_path, "remesh_footwear")
            advanced_panel.separator()
            body_box = advanced_panel.box()
            _draw_group_title(body_box, "Character Extras", icon="MODIFIER")
            _draw_toggle_prop(body_box, remesh, remesh_path, "remesh_face_head")
            _draw_toggle_prop(body_box, remesh, remesh_path, "remesh_tail")
            _draw_toggle_prop(body_box, remesh, remesh_path, "remesh_wings")
            advanced_panel.separator()
            other_box = advanced_panel.box()
            _draw_group_title(other_box, "Other", icon="OBJECT_DATA")
            _draw_toggle_prop(other_box, remesh, remesh_path, "remesh_objects")
            _draw_toggle_prop(other_box, remesh, remesh_path, "remesh_unclassified")

        rigging_header, rigging_panel = layout.panel_prop(state, "show_rigging_section")
        _draw_header(rigging_header, "Rigging", icon="ARMATURE_DATA", alert=True)
        if rigging_panel:
            rigging_panel.label(text="Build Armature also binds and smooths weights when Auto Bind On Build is ON.")
            rigging_panel.separator()
            build_button = rigging_panel.row()
            build_button.scale_y = 1.8
            build_button.operator("hallway_avatar.build_armature", icon="ARMATURE_DATA")

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


class HALLWAYAVATAR_PT_import_popover(Panel):
    bl_label = "Import Options"
    bl_idname = "HALLWAYAVATAR_PT_import_popover"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 14

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        state = context.scene.hallway_avatar_state
        _draw_import_settings(layout, state)


class HALLWAYAVATAR_PT_remesh_popover(Panel):
    bl_label = "Main Settings"
    bl_idname = "HALLWAYAVATAR_PT_remesh_popover"
    bl_space_type = "VIEW_3D"
    bl_region_type = "HEADER"
    bl_ui_units_x = 15

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        remesh = context.scene.hallway_avatar_state.qremesh_settings
        _draw_remesh_settings(layout, remesh)


classes = (
    HALLWAYAVATAR_UL_layers,
    HALLWAYAVATAR_PT_import_popover,
    HALLWAYAVATAR_PT_remesh_popover,
    HALLWAYAVATAR_PT_main,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
