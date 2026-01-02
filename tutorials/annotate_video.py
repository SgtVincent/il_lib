import argparse
import json
import os
import cv2
from tqdm import tqdm

def draw_text_with_background(img, text, position, font_scale=0.8, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def draw_progress_bar(img, progress, position, width, height, color=(0, 255, 0), bg_color=(50, 50, 50)):
    x, y = position
    # Draw background
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    # Draw progress
    fill_width = int(width * progress)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)

def annotate_frame(frame, frame_idx, skill_anns, primitive_anns):
    # Find active skill and primitive
    active_skill = None
    skill_progress = 0
    for ann in skill_anns:
        start, end = ann['frame_duration']
        if start <= frame_idx < end:
            active_skill = ann
            skill_progress = (frame_idx - start) / (end - start)
            break
            
    active_primitive = None
    prim_progress = 0
    for ann in primitive_anns:
        start, end = ann['frame_duration']
        if start <= frame_idx < end:
            active_primitive = ann
            prim_progress = (frame_idx - start) / (end - start)
            break
            
    # Annotate
    # Skill
    if active_skill:
        desc = active_skill['skill_description'][0]
        draw_text_with_background(frame, f"Skill: {desc}", (20, 40), bg_color=(0, 0, 150))
        draw_progress_bar(frame, skill_progress, (20, 50), 200, 10, color=(0, 150, 255))
        
    # Primitive
    if active_primitive:
        desc = active_primitive['primitive_description'][0]
        draw_text_with_background(frame, f"Primitive: {desc}", (20, 90), bg_color=(0, 100, 0))
        draw_progress_bar(frame, prim_progress, (20, 100), 200, 10, color=(50, 205, 50))
        
    return frame

def main():
    parser = argparse.ArgumentParser(description="Annotate video with skill and primitive annotations.")
    parser.add_argument("--dataset_path", type=str, default="DATASETS/behavior/2025-challenge-demos", help="Path to the dataset root")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID")
    parser.add_argument("--episode_id", type=int, default=10, help="Episode ID")
    parser.add_argument("--camera_id", type=str, default="head", help="Camera ID (e.g., head, left_wrist, right_wrist)")
    parser.add_argument("--output_path", type=str, default="annotated_video.mp4", help="Path to save the output video")
    
    args = parser.parse_args()
    
    task_name = f"task-{args.task_id:04d}"
    episode_name = f"episode_{args.episode_id:08d}"
    
    annotation_path = os.path.join(args.dataset_path, "annotations", task_name, f"{episode_name}.json")
    video_path = os.path.join(args.dataset_path, "videos", task_name, f"observation.images.rgb.{args.camera_id}", f"{episode_name}.mp4")
    
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return
        
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    print(f"Loading annotations from {annotation_path}")
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
        
    skill_annotations = annotations.get("skill_annotation", [])
    primitive_annotations = annotations.get("primitive_annotation", [])
    
    print(f"Processing video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    print(f"Saving annotated video to {args.output_path}")
    
    for frame_idx in tqdm(range(total_frames), desc="Annotating Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = annotate_frame(frame, frame_idx, skill_annotations, primitive_annotations)
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()
