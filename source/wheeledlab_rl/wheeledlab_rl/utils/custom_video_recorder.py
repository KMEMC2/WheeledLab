from __future__ import annotations
import inspect
import os
from typing import Callable

import av

import gymnasium as gym
import numpy as np
import wandb
from gymnasium import logger
from gymnasium.core import ActType, ObsType
try:
    # gymnasium>=1.0 moved RecordVideo here.
    from gymnasium.wrappers.rendering import RecordVideo
except ModuleNotFoundError:
    # gymnasium<1.0 (e.g. Isaac Sim's bundled 0.29.1) still exposes it at the package level.
    from gymnasium.wrappers import RecordVideo




class CustomRecordVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        enable_wandb: bool = True,
        video_resolution: tuple[int, int] = (1280, 720),
        video_crf: int = 30,
        num_envs_to_render: int | None = None,
        env_indices: list[int] | None = None,
        grid_shape: tuple[int, int] | None = None,
    ):
        if enable_wandb and wandb.run.name is None:
            raise ValueError("wandb must be initialized before wrapping.")

        record_video_kwargs = dict(
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            fps=fps,
            disable_logger=disable_logger,
        )
        if "fps" not in inspect.signature(RecordVideo.__init__).parameters:
            # gymnasium<1.0 (e.g. Isaac Sim's bundled 0.29.1) has no fps parameter.
            del record_video_kwargs["fps"]
        super().__init__(env, video_folder, **record_video_kwargs)
        self.enable_wandb = enable_wandb
        self.video_resolution = video_resolution
        self.video_crf = video_crf
        self.num_envs_to_render = num_envs_to_render
        self.env_indices = env_indices
        self.grid_shape = grid_shape

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            # Determine whether frames contain multiple envs per timestep
            first = self.recorded_frames[0]
            arr_like = None
            try:
                arr_like = np.asarray(first)
            except Exception:
                arr_like = None

            if arr_like is not None and arr_like.ndim == 4:
                # Shape assumed (num_envs, H, W, C)
                total_envs = arr_like.shape[0]
                n_to_save = (
                    min(self.num_envs_to_render, total_envs)
                    if self.num_envs_to_render is not None
                    else total_envs
                )

                # determine which env indices to include in the grid
                if self.env_indices is not None:
                    indices = [i for i in self.env_indices if 0 <= i < total_envs]
                    indices = indices[:n_to_save]
                else:
                    indices = list(range(n_to_save))

                if len(indices) == 0:
                    # fallback to single combined video if no valid indices
                    indices = [0]

                # determine grid shape (rows, cols)
                n = len(indices)
                if self.grid_shape is not None:
                    rows, cols = self.grid_shape
                else:
                    rows = int(np.ceil(np.sqrt(n)))
                    cols = int(np.ceil(n / rows))

                # target size for each cell
                cell_w, cell_h = self.video_resolution[0], self.video_resolution[1]
                combined_w = cols * cell_w
                combined_h = rows * cell_h

                path = os.path.join(self.video_folder, f"{self._video_name}_grid.mp4")
                output = av.open(path, "w")
                output_stream = output.add_stream(
                    "libx264",
                    rate=round(self.frames_per_sec),
                )
                output_stream.width, output_stream.height = (combined_w, combined_h)
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": str(self.video_crf), "preset": "veryslow"}

                for frame in self.recorded_frames:
                    frame_arr = np.asarray(frame)
                    # build list of resized env frames
                    cell_imgs = []
                    for idx in indices:
                        env_frame = frame_arr[idx]
                        # convert to av frame and reformat to target cell size, then back to ndarray
                        vf = av.VideoFrame.from_ndarray(env_frame, format="rgb24")
                        vf = vf.reformat(width=cell_w, height=cell_h)
                        env_resized = vf.to_ndarray(format="rgb24")
                        cell_imgs.append(env_resized)

                    # pad with black frames if necessary
                    while len(cell_imgs) < rows * cols:
                        cell_imgs.append(np.zeros_like(cell_imgs[0]))

                    # tile into combined image
                    combined = np.zeros((combined_h, combined_w, 3), dtype=cell_imgs[0].dtype)
                    for i, img in enumerate(cell_imgs):
                        r = i // cols
                        c = i % cols
                        y0 = r * cell_h
                        x0 = c * cell_w
                        combined[y0 : y0 + cell_h, x0 : x0 + cell_w] = img

                    video_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
                    packet = output_stream.encode(video_frame)
                    output.mux(packet)

                packet = output_stream.encode()
                output.mux(packet)
                output.close()
                if self.enable_wandb:
                    wandb.log({"Video/grid": wandb.Video(path)}, commit=False)
            else:
                path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
                output = av.open(path, "w")
                output_stream = output.add_stream(
                    "libx264",
                    rate=round(self.frames_per_sec),
                )
                output_stream.width, output_stream.height = self.video_resolution
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": str(self.video_crf), "preset": "veryslow"}
                for frame in self.recorded_frames:
                    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    video_frame = video_frame.reformat(
                        width=self.video_resolution[0], height=self.video_resolution[1]
                    )
                    packet = output_stream.encode(video_frame)
                    output.mux(packet)
                packet = output_stream.encode()
                output.mux(packet)
                output.close()
                if self.enable_wandb:
                    wandb.log({"Video": wandb.Video(path)}, commit=False)

        self.recorded_frames = []
        self.recording = False
        self._video_name = None