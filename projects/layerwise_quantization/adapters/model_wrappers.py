"""Wrappers for SAM2 models to expose feature extraction with student/teacher API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

__all__ = ["TeacherStudentPair", "build_teacher_student"]


@dataclass
class TeacherStudentPair:
    teacher: SAM2ImagePredictor
    student: SAM2ImagePredictor


def _ensure_predictor(model_config: str, checkpoint: Optional[str], device: torch.device) -> SAM2ImagePredictor:
    model = build_sam2(
        config_file=model_config,
        ckpt_path=checkpoint,
        device=str(device),
        mode="eval",
    )
    predictor = SAM2ImagePredictor(model)
    return predictor


def build_teacher_student(
    teacher_config: str,
    teacher_ckpt: Optional[str],
    student_config: str,
    student_ckpt: Optional[str],
    device: torch.device,
) -> TeacherStudentPair:
    teacher = _ensure_predictor(teacher_config, teacher_ckpt, device)
    student = _ensure_predictor(student_config, student_ckpt, device)
    return TeacherStudentPair(teacher=teacher, student=student)
