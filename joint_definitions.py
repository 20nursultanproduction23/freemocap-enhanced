import numpy as np

BODY_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
NUM_BODY = len(BODY_LANDMARK_NAMES)  # 33
NUM_RIGHT_HAND = 21
NUM_LEFT_HAND = 21
NUM_FACE = 478
NUM_TOTAL = NUM_BODY + NUM_RIGHT_HAND + NUM_LEFT_HAND + NUM_FACE  # 553

BODY_IDX = {name: i for i, name in enumerate(BODY_LANDMARK_NAMES)}

BONE_CONNECTIONS = [
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]

VIRTUAL_MARKERS = {
    "head_center": {"left_ear": 0.5, "right_ear": 0.5},
    "neck_center": {"left_shoulder": 0.5, "right_shoulder": 0.5},
    "hips_center": {"left_hip": 0.5, "right_hip": 0.5},
    "trunk_center": {"left_shoulder": 0.25, "right_shoulder": 0.25, "left_hip": 0.25, "right_hip": 0.25},
}

LEFT_FOOT_MARKERS = ["left_ankle", "left_heel", "left_foot_index"]
RIGHT_FOOT_MARKERS = ["right_ankle", "right_heel", "right_foot_index"]


def get_body_slice():
    return slice(0, NUM_BODY)

def get_right_hand_slice():
    return slice(NUM_BODY, NUM_BODY + NUM_RIGHT_HAND)

def get_left_hand_slice():
    return slice(NUM_BODY + NUM_RIGHT_HAND, NUM_BODY + NUM_RIGHT_HAND + NUM_LEFT_HAND)

def get_face_slice():
    return slice(NUM_BODY + NUM_RIGHT_HAND + NUM_LEFT_HAND, NUM_TOTAL)


def compute_bone_lengths(skeleton_data):
    """Compute average bone lengths across all frames.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3) array
        
    Returns:
        dict mapping (parent_name, child_name) -> average_length
    """
    num_frames = skeleton_data.shape[0]
    bone_lengths = {}
    
    for parent_name, child_name in BONE_CONNECTIONS:
        parent_idx = BODY_IDX[parent_name]
        child_idx = BODY_IDX[child_name]
        
        parent_pos = skeleton_data[:, parent_idx, :]  # (numFrames, 3)
        child_pos = skeleton_data[:, child_idx, :]     # (numFrames, 3)
        
        diffs = parent_pos - child_pos
        lengths = np.sqrt(np.sum(diffs ** 2, axis=1))  # (numFrames,)
        
        valid = ~np.isnan(lengths)
        if np.any(valid):
            bone_lengths[(parent_name, child_name)] = np.mean(lengths[valid])
        else:
            bone_lengths[(parent_name, child_name)] = 0.0
    
    return bone_lengths


def compute_foot_center(skeleton_data, foot_markers):
    """Compute center of foot markers.
    
    Args:
        skeleton_data: (numFrames, numTrackedPoints, 3)
        foot_markers: list of landmark names
        
    Returns:
        (numFrames, 3) array of foot center positions
    """
    indices = [BODY_IDX[name] for name in foot_markers]
    foot_data = skeleton_data[:, indices, :]  # (numFrames, len(foot_markers), 3)
    return np.nanmean(foot_data, axis=1)
