"""
Annotation QA Environment — A real-world OpenEnv for ML annotation quality assurance.

This environment uses real COCO val2017 images and challenges a VLM agent
to detect and correct intentional errors in the annotations.
"""

from .client import AnnotationQAEnv
from .models import AnnotationQAAction, AnnotationQAObservation, AnnotationQAState

__all__ = [
    "AnnotationQAEnv",
    "AnnotationQAAction",
    "AnnotationQAObservation",
    "AnnotationQAState",
]
