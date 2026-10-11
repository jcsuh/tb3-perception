"""
좌표 변환 계산 도우미.

URDF와 같은 약속을 쓴다: rpy는 고정축 X(roll), Y(pitch), Z(yaw) 순서로 돌린다.
즉 R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""
import numpy as np


def rpy_to_matrix(roll, pitch, yaw):
    """Roll, pitch, yaw(rad)를 3x3 회전 행렬로 바꾼다."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_rpy(r):
    """3x3 회전 행렬을 roll, pitch, yaw(rad)로 바꾼다 (rpy_to_matrix의 역)."""
    pitch = np.arcsin(-np.clip(r[2, 0], -1.0, 1.0))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def rotation_angle(r):
    """회전 행렬이 나타내는 회전의 크기(rad)를 구한다."""
    c = (np.trace(r) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
