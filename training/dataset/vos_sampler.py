# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass
from typing import List, Optional

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    frames: List[int]
    object_ids: List[int]


class VOSSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()


class RandomUniformSampler(VOSSampler):
    def __init__(
        self,
        num_frames,
        max_num_objects,
        reverse_time_prob=0.0,
    ):
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, segment_loader, epoch=None):

        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + step] for step in range(self.num_frames)]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]

            # Get first frame object ids
            visible_object_ids = []
            loaded_segms = segment_loader.load(frames[0].frame_idx)
            if isinstance(loaded_segms, LazySegments):
                # LazySegments for SA1BRawDataset
                visible_object_ids = list(loaded_segms.keys())
            else:
                for object_id, segment in segment_loader.load(
                    frames[0].frame_idx
                ).items():
                    if segment.sum():
                        visible_object_ids.append(object_id)

            # First frame needs to have at least a target to track
            if len(visible_object_ids) > 0:
                break
            if retry >= MAX_RETRIES - 1:
                raise Exception("No visible objects")

        object_ids = random.sample(
            visible_object_ids,
            min(len(visible_object_ids), self.max_num_objects),
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """Evaluation sampler with optional frame sub-sampling."""

    def __init__(
        self,
        *,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
    ):
        super().__init__()
        self.max_frames = max_frames if max_frames is None or max_frames > 0 else None
        self.frame_stride = max(1, frame_stride)

    def sample(self, video, segment_loader, epoch=None):
        """Sampling frames/objects with optional sub-sampling."""
        if self.sort_frames:
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            frames = video.frames

        if self.frame_stride > 1:
            frames = frames[:: self.frame_stride]
        if self.max_frames is not None:
            frames = frames[: self.max_frames]
        if len(frames) == 0:
            raise Exception("No frames selected for evaluation")

        object_ids = list(segment_loader.load(frames[0].frame_idx).keys())
        if len(object_ids) == 0:
            raise Exception("First frame of the video has no objects")

        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
