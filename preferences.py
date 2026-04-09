from __future__ import annotations

import bpy

from . import utils, wheel_manager


class HallwayAvatarGenPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    auto_import_results: bpy.props.BoolProperty(
        name="Auto Import Results",
        default=True,
        description="Import completed decomposition results into the current Blender scene automatically.",
    )
    plane_scale: bpy.props.FloatProperty(
        name="Plane Scale",
        default=0.005,
        min=0.0001,
        soft_max=0.05,
        description="Viewport size multiplier used when converting 2D layer bounds into image planes.",
    )
    depth_spacing: bpy.props.FloatProperty(
        name="Depth Spacing",
        default=0.05,
        min=0.001,
        soft_max=1.0,
        description="Distance between imported planes based on their inferred depth ordering.",
    )
    default_device: bpy.props.EnumProperty(
        name="Preferred Device",
        items=(
            ("auto", "Auto", "Pick CUDA first, then Apple Metal, then CPU."),
            ("cuda", "CUDA", "Force NVIDIA CUDA when available."),
            ("mps", "Apple Metal", "Force Apple Metal when available."),
            ("cpu", "CPU", "Force CPU inference."),
        ),
        default="auto",
    )
    default_quant_mode: bpy.props.EnumProperty(
        name="Quantization",
        items=(
            ("auto", "Auto", "Use NF4 on CUDA and full precision elsewhere."),
            ("nf4", "NF4", "Prefer NF4 4-bit inference when supported."),
            ("none", "Full Precision", "Disable NF4 and run standard weights."),
        ),
        default="auto",
    )
    default_resolution: bpy.props.IntProperty(
        name="Default Resolution",
        default=1024,
        min=512,
        max=2048,
        step=64,
    )

    runtime_details_expanded: bpy.props.BoolProperty(
        name="Runtime Details Expanded",
        default=False,
        options={"SKIP_SAVE"},
    )

    advanced_debug_expanded: bpy.props.BoolProperty(
        name="Advanced Debug Expanded",
        default=False,
        options={"SKIP_SAVE"},
    )

    installer_details_expanded: bpy.props.BoolProperty(
        name="Installer Details Expanded",
        default=False,
        options={"SKIP_SAVE"},
    )

    def draw(self, context) -> None:
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        def draw_wrapped(box, text: str) -> None:
            for line in utils.wrap_text_to_panel(text, context, full_width=True).splitlines() or [""]:
                box.label(text=line)

        snapshot = wheel_manager.preferences_snapshot()
        install_state = wheel_manager.install_state_snapshot()
        legacy_addons = snapshot["legacy_addons"]
        all_groups_ready = bool(snapshot["groups"]) and all(group_info["installed"] for group_info in snapshot["groups"])

        runtime_box = layout.box()
        runtime_header = runtime_box.row(align=True)
        runtime_header.label(text="Runtime")
        runtime_header.operator("hallway_avatar_gen.rescan_dependencies", text="Rescan", icon="FILE_REFRESH")
        runtime_box.prop(self, "auto_import_results")
        runtime_box.prop(self, "plane_scale")
        runtime_box.prop(self, "depth_spacing")
        runtime_box.prop(self, "default_device")
        runtime_box.prop(self, "default_quant_mode")
        runtime_box.prop(self, "default_resolution")
        if all_groups_ready:
            summary_row = runtime_box.row()
            summary_row.label(text="All requirements detected.", icon="CHECKMARK")
        else:
            summary_row = runtime_box.row()
            summary_row.alert = True
            summary_row.label(text="Some requirements not detected. Please install below.", icon="ERROR")

        runtime_details_row = runtime_box.row(align=True)
        runtime_details_row.alignment = "LEFT"
        runtime_details_icon = "TRIA_DOWN" if self.runtime_details_expanded else "TRIA_RIGHT"
        runtime_details_row.prop(self, "runtime_details_expanded", text="Runtime Details", emboss=False, icon=runtime_details_icon)
        if self.runtime_details_expanded:
            draw_wrapped(runtime_box, f"Detected torch runtime: {snapshot['torch_summary']}")
            if snapshot["torch_origin"]:
                draw_wrapped(runtime_box, f"Torch source: {snapshot['torch_origin']}")
            draw_wrapped(runtime_box, f"Detected pywebview runtime: {snapshot['webview_summary']}")
            if snapshot["webview_origin"]:
                draw_wrapped(runtime_box, f"pywebview source: {snapshot['webview_origin']}")
            for path in snapshot["shared_dependency_paths"]:
                draw_wrapped(runtime_box, f"Using shared deps: {path}")

        deps_box = layout.box()
        deps_box.label(text="Dependency Wheels")
        draw_wrapped(
            deps_box,
            "If a dependency is already importable from Blender, this extension, or a shared add-on vendor path, you do not need to install it here. Install only copies missing packages into Hallway Avatar Gen's local vendor folder, and it reuses shared wheel caches before downloading anything.",
        )
        host_profile = snapshot["host_profile"]
        deps_box.label(text=f"Current host: {host_profile['system_label']} | {host_profile['backend_label']}")
        for note in host_profile["notes"]:
            deps_box.label(text=note)
        for group_info in snapshot["groups"]:
            group = group_info["group"]
            installed = group_info["installed"]
            status_text = group_info["status_text"]
            group_box = deps_box.box()
            row = group_box.row(align=True)
            row.label(text=group.label, icon="CHECKMARK" if installed else "IMPORT")
            if installed:
                row.label(text="Ready")
            else:
                op = row.operator("hallway_avatar_gen.install_dependency_group", text="Install")
                op.group_key = group.key

            state_matches_group = install_state["group_key"] == group.key and (
                install_state["is_running"] or install_state["message"] != "Idle"
            )
            if state_matches_group:
                status_card = group_box.box()
                primary_line = install_state["current_line"] or install_state["message"]
                if install_state["is_running"]:
                    progress_factor = max(0.02, min(0.98, float(install_state["progress"])))
                    status_card.progress(factor=progress_factor, text="Installing...", translate=False)
                draw_wrapped(status_card, primary_line)

                failed_install = (not install_state["is_running"]) and install_state["last_return_code"] not in (None, 0)
                failure_summary = install_state["failure_summary"] or install_state["message"]
                if failed_install and failure_summary:
                    alert_box = status_card.box()
                    alert_box.alert = True
                    alert_box.label(text="Install Error", icon="ERROR")
                    draw_wrapped(alert_box, failure_summary)

                details_row = status_card.row(align=True)
                details_row.alignment = "LEFT"
                details_icon = "TRIA_DOWN" if self.installer_details_expanded else "TRIA_RIGHT"
                details_row.prop(self, "installer_details_expanded", text="Details", emboss=False, icon=details_icon)
                if self.installer_details_expanded:
                    if install_state["log_path"]:
                        draw_wrapped(status_card, install_state["log_path"])
                    for detail_line in install_state["log_lines"]:
                        draw_wrapped(status_card, detail_line)

            for line in (utils.wrap_text_to_panel(group.description, context, full_width=True).splitlines() or [""]):
                group_box.label(text=line)
            if not state_matches_group:
                group_box.label(text=status_text)

        if snapshot["shared_wheel_count"]:
            deps_box.label(text=f"Shared wheel caches: {snapshot['shared_wheel_count']} found")

        debug_box = layout.box()
        debug_row = debug_box.row(align=True)
        debug_row.alignment = "LEFT"
        debug_icon = "TRIA_DOWN" if self.advanced_debug_expanded else "TRIA_RIGHT"
        debug_row.prop(self, "advanced_debug_expanded", text="Advanced Debug", emboss=False, icon=debug_icon)
        if self.advanced_debug_expanded:
            if legacy_addons:
                debug_notice = debug_box.box()
                debug_notice.alert = True
                debug_notice.label(text="Legacy Add-on Notice", icon="ERROR")
                draw_wrapped(
                    debug_notice,
                    "A legacy Hallway add-on is still installed in Blender's scripts/addons folder and is responsible for the pip startup errors in blender_startup.log.",
                )
                for legacy_path in legacy_addons:
                    draw_wrapped(debug_notice, f"Legacy add-on path: {legacy_path}")
            else:
                debug_box.label(text="No legacy Hallway add-on detected.", icon="CHECKMARK")

        actions = layout.row(align=True)
        actions.operator("hallway_avatar_gen.launch_webview_ui", text="Launch UI")
        actions.operator("hallway_avatar_gen.open_runtime_folder", text="Open Runtime Folder")
