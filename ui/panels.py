from __future__ import annotations

import bpy
from bpy.types import Panel, UIList

from ..utils import env


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
