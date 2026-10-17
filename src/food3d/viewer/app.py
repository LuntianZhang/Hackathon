from dataclasses import asdict
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..io import read_json, write_json
from ..schemas import CameraFrame, UNKNOWN_ID, load_scene
from ..synthetic import look_at
from ..volume import calibrate, estimate_bundle, load_calibration, volume_text
from .renderer import GaussianRenderer, resize_camera
from .orbit import drag_camera


def describe_click(labels: np.ndarray, confidence: np.ndarray, x: int, y: int, names: dict[int, str]) -> tuple[int, str]:
    if not 0 <= x < labels.shape[1] or not 0 <= y < labels.shape[0]:
        return 0, "点击位置超出画面"
    instance_id = int(labels[y, x])
    if instance_id == 0:
        return 0, "背景（Instance 0）"
    if instance_id == UNKNOWN_ID or confidence[y, x] < 0.55:
        return 0, "该位置的三维语义证据不足"
    return instance_id, f"Instance {instance_id}：{names.get(instance_id, '未命名')} · 投票置信度 {confidence[y,x]:.2f}"


def create_app(bundle: Path):
    import gradio as gr
    renderer = GaussianRenderer(bundle)
    if renderer.features is None:
        raise ValueError("请先运行三维语义融合，再打开实例点击界面")
    _, frames = load_scene(bundle / "scene.json")
    instance_file = bundle / "semantics" / "instances.json"
    volume_report = estimate_bundle(bundle)
    def measurement(selected):
        return volume_text(volume_report, selected, load_calibration(bundle, volume_report))
    def calibration_note():
        value = load_calibration(bundle, volume_report)
        if value is None:
            return "尚未标定：体积显示为模型单位³。"
        if value.get('source')=='pansense_aruco_board_multiview':
            return f"已使用{value['board_name']}估算体积尺度，{len(value['inlier_views'])} 个视角通过校验。"
        if value.get('source')=='white_paper_rectangle_multiview':
            return '已使用历史纸片标定尺度。'
        return (f"已使用 Instance {value['reference_instance_id']} 的最大跨度 "
                f"{value['measured_span_cm']:g} cm 标定整个场景。")
    def set_scale(selected, measured_span_cm):
        try:
            calibrate(bundle, volume_report, selected, float(measured_span_cm))
            return measurement(selected), calibration_note()
        except (ValueError, TypeError) as error:
            return measurement(selected), str(error)
    def clear_scale(selected):
        write_json(bundle / "measurements" / "calibration.json", {"status": "disabled"})
        return measurement(selected), calibration_note()
    def names():
        return {int(entry["instance_id"]): entry["name"] for entry in read_json(instance_file)["instances"]}
    semantic_food = (renderer.semantic_ids > 0) & (renderer.semantic_ids != UNKNOWN_ID)
    points = renderer.model.means[semantic_food]
    if len(points) == 0:
        raise ValueError("融合结果没有可靠食物实例，无法构建演示")
    target = np.median(points, axis=0)
    positions = np.array([np.asarray(frame.camera_to_world)[:3, 3] for frame in frames])
    radius = float(np.median(np.linalg.norm(positions-target, axis=1)))
    reference_frame = read_json(instance_file).get("reference_frame")
    default = next((frame for frame in frames if frame.frame_id == reference_frame), frames[0])
    default = resize_camera(default, 640)
    initial_offset = np.asarray(default.camera_to_world)[:3, 3] - target
    initial_azimuth = float(np.degrees(np.arctan2(initial_offset[1], initial_offset[0])) % 360)
    initial_elevation = float(np.clip(np.degrees(np.arctan2(initial_offset[2], np.linalg.norm(initial_offset[:2]))), 10, 80))
    initial_distance = float(np.clip(np.linalg.norm(initial_offset) / radius, 0.6, 1.8))
    first = renderer.render(default)
    def view_state(camera, result, controller=None):
        state = {"camera": asdict(camera), "labels": result["instance_ids"], "confidence": result["confidence"]}
        state.update({key: controller[key] for key in ["base_camera", "orbit", "epoch", "ack"]} if controller else
                     {"base_camera": asdict(camera), "orbit": {"yaw": 0, "pitch": 0, "zoom": 1}, "epoch": uuid4().hex, "ack": 0})
        return state
    def control_value(state):
        return json.dumps({"epoch": state["epoch"], "ack": state["ack"], **state["orbit"]})
    first_state = view_state(default, first)
    def display(camera: CameraFrame, selected: int, controller=None):
        result = renderer.render(camera, selected)
        state = view_state(camera, result, controller)
        return result["rgb"], state, control_value(state)
    def captured(frame_id, selected):
        camera = next(frame for frame in frames if frame.frame_id == frame_id)
        return display(resize_camera(camera, 640), selected)
    def orbit(azimuth, elevation, distance, selected):
        a, e = np.deg2rad(azimuth), np.deg2rad(elevation)
        position = target + radius*distance*np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
        camera = CameraFrame("orbit", "", 640, 480, [[600, 0, 319.5], [0, 600, 239.5], [0, 0, 1]], look_at(position, target).tolist())
        return display(camera, selected)
    def click(state, evt: gr.SelectData):
        x, y = evt.index
        selected, text = describe_click(state["labels"], state["confidence"], int(x), int(y), names())
        rgb, new_state, controls = display(CameraFrame(**state["camera"]), selected, state)
        return rgb, new_state, selected, text, measurement(selected), controls
    def drag(payload, state, selected):
        gesture = json.loads(payload)
        if gesture.get("epoch") != state["epoch"] or int(gesture["seq"]) <= state["ack"]:
            return gr.skip(), gr.skip(), gr.skip()
        controller = dict(state)
        controller["ack"] = int(gesture["seq"])
        controller["orbit"] = {key: float(gesture[key]) for key in ["yaw", "pitch", "zoom"]}
        camera = drag_camera(CameraFrame(**state["base_camera"]), target, **controller["orbit"])
        return display(camera, selected, controller)
    def rename(selected, new_name):
        if selected <= 0 or selected == UNKNOWN_ID or not new_name.strip():
            return "请先选中食物并输入名称"
        data = read_json(instance_file)
        for entry in data["instances"]:
            if int(entry["instance_id"]) == selected:
                entry["name"] = new_name.strip()
                entry["name_source"] = "manual_correction"
                write_json(instance_file, data)
                return f"Instance {selected}：{entry['name']}（名称已保存）"
        return "实例不存在"
    css = ("#food3d_gesture, #food3d_camera_control {display:none !important;} "
           "#food3d_scene img {cursor:grab; touch-action:none; user-select:none;} "
           "#food3d_scene.food3d_rotating img {cursor:grabbing;}")
    js = Path(__file__).with_name("drag.js").read_text(encoding="utf-8")
    with gr.Blocks(title="Food3D · 食物三维实例", css=css, js=js) as app:
        gr.Markdown("# 食物三维实例\n按住画面拖动旋转，滚轮缩放；轻点食物查看名称和体积。换视角后，选中的实例会继续高亮。")
        current = gr.State(first_state)
        selected = gr.State(0)
        gesture_bridge = gr.Textbox(value="", elem_id="food3d_gesture", show_label=False, container=False)
        camera_control = gr.Textbox(value=control_value(first_state), elem_id="food3d_camera_control", show_label=False, container=False, interactive=False)
        with gr.Row():
            image = gr.Image(value=first["rgb"], type="numpy", label="三维场景 · 拖动旋转 / 滚轮缩放", interactive=False, height=480, elem_id="food3d_scene")
            with gr.Column():
                description = gr.Textbox(label="选中食物", value="点击食物查看 Instance ID 和名称", interactive=False)
                volume = gr.Textbox(label="体积粗估（凸包包围体积）", value=measurement(0), lines=4, interactive=False)
                captured_view = gr.Dropdown(choices=[frame.frame_id for frame in frames], value=default.frame_id, label="拍摄视角")
                reset_view = gr.Button("恢复拍摄视角")
                with gr.Accordion("精细视角设置（可选）", open=False):
                    azimuth = gr.Slider(0, 360, value=initial_azimuth, step=2, label="环绕角度")
                    elevation = gr.Slider(10, 80, value=initial_elevation, step=2, label="俯视角度")
                    distance = gr.Slider(0.6, 1.8, value=initial_distance, step=0.05, label="观察距离")
                    update = gr.Button("更新环绕视角")
                new_name = gr.Textbox(label="修正名称")
                save = gr.Button("保存名称")
        with gr.Accordion("尺寸标定：将体积换算为毫升", open=False):
            gr.Markdown("选中一个食物，输入实际测量的最大两点直线距离（例如香蕉两端的直线跨度）。"
                        "标定对整个场景生效，并保存在本机。渲染图片没有真实尺寸时，请保持未标定。"
                        "\n\n这是三维包围体积粗估，尚未通过真实体积测量验证。")
            measured_span = gr.Number(label="选中食物实际最大跨度（厘米）", value=None, minimum=0)
            calibration_status = gr.Textbox(label="标定状态", value=calibration_note(), interactive=False)
            with gr.Row():
                apply_scale = gr.Button("用选中食物标定")
                reset_scale = gr.Button("清除尺寸标定")
        render_outputs = [image, current, camera_control]
        render_options = {"concurrency_limit": 1, "concurrency_id": "gaussian_render", "show_progress": "hidden"}
        gesture_bridge.input(drag, [gesture_bridge, current, selected], render_outputs, trigger_mode="always_last", **render_options)
        image.select(click, [current], [image, current, selected, description, volume, camera_control], **render_options)
        captured_view.change(captured, [captured_view, selected], render_outputs, **render_options)
        reset_view.click(captured, [captured_view, selected], render_outputs, **render_options)
        update.click(orbit, [azimuth, elevation, distance, selected], render_outputs, **render_options)
        save.click(rename, [selected, new_name], [description])
        apply_scale.click(set_scale, [selected, measured_span], [volume, calibration_status], concurrency_limit=1)
        reset_scale.click(clear_scale, [selected], [volume, calibration_status], concurrency_limit=1)
    return app


def serve(bundle: Path, host: str = "127.0.0.1", port: int = 7860) -> None:
    from food3d.paths import ProjectPaths
    os.environ.setdefault("GRADIO_TEMP_DIR", str(ProjectPaths().cache_root / 'gradio'))
    create_app(bundle.resolve()).queue(default_concurrency_limit=1).launch(server_name=host, server_port=port, share=False)
