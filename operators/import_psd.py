from __future__ import annotations

import os

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from .. import properties
from ..core import alpha_mesh_adapter, facial_video_preview, mtoon_materials, part_classifier, pipeline, psd_io, qremeshify, weighting
from ..utils import blender as blender_utils
from ..utils.logging import get_logger


logger = get_logger("import_psd_operator")


class HALLWAYAVATAR_OT_import_psd(Operator, ImportHelper):
    bl_idname = "hallway_avatar.import_psd"
    bl_label = "Import PSD Avatar"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".psd"
    filter_glob: StringProperty(default="*.psd;*.PSD", options={"HIDDEN"})

    _timer = None
    _parts = None
    _part_index = 0
    _imported_objects = None
    _collection = None
    _remesh_candidates = None
    _remesh_index = 0
    _remeshed_count = 0
    _remesh_settings = None
    _stage = ""
    _import_report = ""
    _rig_plan = None
    _armature_name = ""

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.source_psd_path or "").strip()
        if stored_path:
            resolved = bpy.path.abspath(stored_path)
            if resolved.lower().endswith(".psd") and os.path.isfile(resolved):
                self.filepath = resolved
                return self.execute(context)
            self.filepath = resolved

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context):
        state = context.scene.hallway_avatar_state
        filepath = (self.filepath or state.source_psd_path or "").strip()
        if not filepath:
            self.report({"ERROR"}, "Choose a PSD first.")
            return {"CANCELLED"}
        state.source_psd_path = filepath
        if self._facial_video_inputs_required(state):
            state.last_report = "Select Facial config txt, Facial Video, and Mouth Video if mouth plane is ON before importing the PSD Avatar."
            self._show_facial_video_inputs_popup(context)
            return {"CANCELLED"}
        if context.window is None:
            return self._execute_blocking(context, filepath)
        return self._start_modal(context, filepath)

    @staticmethod
    def _facial_video_inputs_required(state) -> bool:
        if not state.auto_setup_facial_video:
            return False
        if not (state.facial_video_transform_path or "").strip() or not (state.facial_video_path or "").strip():
            return True
        return bool(state.setup_mouth_video_plane) and not (state.mouth_video_path or "").strip()

    @staticmethod
    def _show_facial_video_inputs_popup(context: bpy.types.Context) -> None:
        if bpy.app.background:
            return
        window_manager = getattr(context, "window_manager", None)
        if window_manager is None:
            return

        def draw(self, _context):
            layout = self.layout
            layout.label(text="Facial Video Preview is ON.")
            layout.label(text="Select Facial config txt and video path(s) FIRST.")
            layout.label(text="If Mouth Video Plane is ON, select a Mouth Video too.")
            layout.label(text="The PSD path was saved; import after both are set.")

        try:
            window_manager.popup_menu(draw, title="Facial Video Inputs Required", icon="ERROR")
        except Exception:
            pass

    def modal(self, context: bpy.types.Context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            done = self._step(context)
        except Exception as exc:
            logger.exception("PSD import modal failed during stage %s", self._stage)
            self._finish_modal(context, cancelled=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if done:
            self._finish_modal(context)
            self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def _execute_blocking(self, context: bpy.types.Context, filepath: str):
        try:
            pipeline.import_psd_scene(context, filepath)
            self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
            return {"FINISHED"}
        except Exception as exc:
            logger.exception("PSD import failed in blocking mode")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _start_modal(self, context: bpy.types.Context, filepath: str):
        self.filepath = filepath
        self._parts = []
        self._part_index = 0
        self._imported_objects = []
        self._collection = None
        self._remesh_candidates = []
        self._remesh_index = 0
        self._remeshed_count = 0
        self._remesh_settings = None
        self._stage = "parse"
        self._import_report = ""
        self._rig_plan = None
        self._armature_name = ""

        context.window_manager.progress_begin(0, 100)
        state = context.scene.hallway_avatar_state
        state.import_progress_visible = True
        state.import_progress = 0.0
        state.import_progress_text = "Starting PSD avatar import..."
        state.last_report = state.import_progress_text
        state.remesh_performed = False
        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_modal(self, context: bpy.types.Context, *, cancelled: bool = False) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        if cancelled:
            context.scene.hallway_avatar_state.import_progress_visible = False
            context.scene.hallway_avatar_state.import_progress_text = "PSD avatar import cancelled"
            context.scene.hallway_avatar_state.last_report = "PSD avatar import cancelled"
        else:
            context.scene.hallway_avatar_state.import_progress_visible = False
            context.scene.hallway_avatar_state.import_progress = 0.0
            context.scene.hallway_avatar_state.import_progress_text = ""
        self._tag_viewports(context)

    def _progress(self, context: bpy.types.Context, value: int, report: str) -> None:
        clamped_value = max(0, min(100, value))
        context.window_manager.progress_update(clamped_value)
        state = context.scene.hallway_avatar_state
        state.import_progress_visible = True
        state.import_progress = clamped_value / 100.0
        state.import_progress_text = report
        state.last_report = report
        self._tag_viewports(context)

    @staticmethod
    def _tag_viewports(context: bpy.types.Context) -> None:
        screen = getattr(context, "screen", None)
        if screen is None:
            return
        for area in screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES", "OUTLINER"}:
                area.tag_redraw()

    def _step(self, context: bpy.types.Context) -> bool:
        state = context.scene.hallway_avatar_state

        if self._stage == "parse":
            cache_dir = pipeline._cache_dir_from_context(context)
            self._parts = psd_io.load_psd_layer_parts(
                self.filepath,
                ignore_hidden_layers=state.ignore_hidden_layers,
                ignore_empty_layers=state.ignore_empty_layers,
                min_visible_pixels=state.min_visible_pixels,
                alpha_noise_floor=state.alpha_noise_floor,
                visible_alpha_threshold=state.visible_alpha_threshold,
                auto_alpha_threshold_boost=state.auto_alpha_threshold_boost,
                keep_tiny_named_parts=state.keep_tiny_named_parts,
                configured_cache_dir=cache_dir,
            )
            part_classifier.classify_parts(self._parts)
            pipeline._skip_facial_features_when_disabled(self._parts, state.import_facial_features)
            self._collection = (
                blender_utils.clear_collection(state.imported_collection_name)
                if state.replace_existing
                else blender_utils.ensure_collection(state.imported_collection_name)
            )
            self._part_index = 0
            self._stage = "import_layers"
            self._progress(context, 8, f"Parsed PSD: {len(self._parts)} layers")
            return False

        if self._stage == "import_layers":
            parts = self._parts or []
            if self._part_index >= len(parts):
                self._stage = "stack_layers"
                self._progress(context, 45, f"Imported {len(self._imported_objects or [])} mesh layers")
                return False

            part = parts[self._part_index]
            self._part_index += 1
            if part.skipped or part.area <= 0 or part.local_alpha_bbox[2] <= part.local_alpha_bbox[0] or part.local_alpha_bbox[3] <= part.local_alpha_bbox[1]:
                part.skipped = True
                if not part.skip_reason:
                    part.skip_reason = "empty alpha after rasterization"
            else:
                obj = alpha_mesh_adapter.build_layer_mesh(
                    context,
                    part,
                    self._collection,
                    grid_resolution=state.mesh_grid_resolution,
                    trace_contrast_remap=(state.trace_contrast_low, state.trace_contrast_high),
                )
                part.imported_object_name = obj.name
                obj["hallway_avatar_semantic_label"] = part.semantic_label
                obj["hallway_avatar_side_guess"] = part.side_guess
                obj["hallway_avatar_confidence"] = part.confidence
                self._imported_objects.append(obj)

            progress = 8 + int(37 * (self._part_index / max(1, len(parts))))
            self._progress(context, progress, f"Importing PSD layers {self._part_index}/{len(parts)}")
            context.view_layer.update()
            return False

        if self._stage == "stack_layers":
            parts = self._parts or []
            imported_objects = self._imported_objects or []
            pipeline._apply_layer_depth_stack([part for part in parts if not part.skipped], imported_objects)
            context.view_layer.update()
            self._stage = "remesh_setup"
            self._progress(context, 48, "Layer stack complete")
            return False

        if self._stage == "remesh_setup":
            parts = self._parts or []
            if state.qremeshify_settings.auto_on_import and self._imported_objects:
                settings = qremeshify.QRemeshifySettings.from_scene_state(state)
                self._remesh_candidates = [
                    part
                    for part in parts
                    if not part.skipped
                    and part.imported_object_name
                    and qremeshify._should_remesh_part(part, settings)
                ]
                self._remesh_settings = settings
                self._remesh_index = 0
                self._stage = "remesh_layers"
                self._progress(context, 50, f"Preparing quad remesh for {len(self._remesh_candidates)} layers")
            else:
                self._stage = "import_transform"
                self._progress(context, 72, "Auto remesh skipped")
            return False

        if self._stage == "remesh_layers":
            candidates = self._remesh_candidates or []
            if self._remesh_index >= len(candidates):
                self._stage = "import_transform"
                self._progress(context, 76, f"Quad remeshed {self._remeshed_count} layers")
                return False

            self._progress(context, 50, f"Quad remeshing {len(candidates)} layers in parallel")
            self._remeshed_count = qremeshify.remesh_parts(context, candidates, self._remesh_settings)
            self._remesh_index = len(candidates)
            self._progress(context, 76, f"Quad remeshed {self._remeshed_count} layers")
            context.view_layer.update()
            return False

        if self._stage == "import_transform":
            parts = self._parts or []
            imported_objects = pipeline._imported_mesh_objects_for_parts(parts)
            z_offset = pipeline._apply_import_geometry_transform(imported_objects)
            context.view_layer.update()
            if abs(z_offset) > 1e-9:
                logger.info(
                    "Modal import scaled layers by %.6f and lifted by %.6fm to Z=%.5f",
                    pipeline.IMPORT_VERTEX_SCALE,
                    z_offset,
                    pipeline.IMPORT_TARGET_MIN_Z,
                )
            self._stage = "mtoon"
            self._progress(context, 78, "Import transform complete")
            return False

        if self._stage == "mtoon":
            parts = self._parts or []
            mtoon_count = mtoon_materials.configure_avatar_mtoon_materials(parts)
            state.source_psd_path = self.filepath
            properties.set_layer_items(context.scene, parts)
            state.remeshed_count = self._remeshed_count
            state.remesh_performed = self._remeshed_count > 0
            if state.qremeshify_settings.auto_on_import and self._imported_objects:
                self._import_report = f"Imported {state.imported_count} layers, remeshed {state.remeshed_count}, skipped {state.skipped_count}"
            else:
                state.remeshed_count = 0
                state.remesh_performed = False
                self._import_report = f"Imported {state.imported_count} layers, skipped {state.skipped_count}"
            state.last_report = self._import_report
            logger.info("Configured MToon material settings on %s imported layer materials", mtoon_count)
            if state.auto_setup_facial_video:
                try:
                    face_video_obj = facial_video_preview.setup_from_state(context, parts=parts, raise_on_missing=False)
                    if face_video_obj is not None:
                        logger.info("Configured facial video preview on %s", face_video_obj.name)
                except Exception as exc:
                    logger.exception("Facial video preview setup failed during PSD import")
                    state.last_report = f"{self._import_report}; facial video preview failed: {exc}"
            if state.auto_rig_on_import and self._imported_objects:
                self._stage = "rig"
                self._progress(context, 82, "MToon setup complete; building rig")
            else:
                self._stage = "finish"
                self._progress(context, 100, self._import_report)
            return False

        if self._stage == "rig":
            armature_obj, rig_plan = pipeline.build_armature_scene(context, bind_weights=False)
            self._armature_name = armature_obj.name
            self._rig_plan = rig_plan
            if state.auto_bind_on_build:
                self._stage = "bind_weights"
                self._progress(context, 90, f"Built {armature_obj.name}; binding weights")
            else:
                state.last_report = f"{self._import_report}; built {armature_obj.name} with {len(rig_plan.bones)} bones"
                self._stage = "finish"
                self._progress(context, 100, state.last_report)
            return False

        if self._stage == "bind_weights":
            armature_obj = bpy.data.objects.get(self._armature_name)
            if armature_obj is None:
                raise RuntimeError("Generated armature disappeared before binding.")
            weighting.bind_parts(context, armature_obj, self._parts or [], rig_plan=self._rig_plan)
            state.last_report = f"{self._import_report}; built {armature_obj.name} with {len(self._rig_plan.bones)} bones"
            self._stage = "finish"
            self._progress(context, 100, state.last_report)
            return False

        return True


classes = (HALLWAYAVATAR_OT_import_psd,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
