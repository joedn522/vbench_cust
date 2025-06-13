import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

from vbench.utils import load_dimension_info

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


class DynamicDegree:
    def __init__(self, args, device, downsample_ratio=1.0):
        self.args = args
        self.device = device
        self.downsample_ratio = downsample_ratio
        self.load_model()
    

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo, scale_factor=1.0):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy() * scale_factor

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, orig_scale, count):
        self.params = {
            "thres": 6.0 * (orig_scale / 256.0),
            "count_num": round(4 * (count / 16.0))
        }


    def infer(self, video_path):
        with torch.no_grad():
            print(f"[DEBUG] infer() got video_path: {video_path}")
            print(f"[DEBUG] os.path.exists(video_path): {os.path.exists(video_path)}")
            print(f"[DEBUG] os.path.isdir(video_path): {os.path.isdir(video_path)}")
            print(f"[DEBUG] os.path.isfile(video_path): {os.path.isfile(video_path)}")
            if video_path.lower().endswith(('.mp4', '.mov')):
                print(f"[DEBUG] video_path ends with .mp4 -> calling get_frames()")
                frames, orig_scale = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                print(f"[DEBUG] video_path is directory -> calling get_frames_from_img_folder()")
                frames, orig_scale = self.get_frames_from_img_folder(video_path)
            else:
                print(f"[ERROR] video_path is neither .mp4 nor directory, raising NotImplementedError")
                raise NotImplementedError
            self.set_params(orig_scale, count=len(frames)) 
            scale_factor = 1.0 / self.downsample_ratio  
            static_score = []
            for image1, image2 in zip(frames[:-1:2], frames[1::2]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up, scale_factor)
                static_score.append(max_rad)
            
            total_score = sum(static_score)
            avg_score = total_score / len(static_score) if static_score else 0

            print(f"[INFO] Total score: {total_score}, Average score: {avg_score}")

            whether_move = self.check_move(static_score)
            return whether_move, total_score, avg_score


    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


    def get_frames(self, video_path):
        """
        Extract frames from the middle 5 seconds of a video file with a specified interval.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: List of frames as PyTorch tensors.
        """
        frame_list = []
        video = cv2.VideoCapture(video_path)

        # Get the FPS and total number of frames of the video
        fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
        interval = max(1, round(fps / 8))  # Set the frame interval

        # Debug log: FPS, total frames, and interval
        print(f"[DEBUG] Video: {video_path}, FPS: {fps}, Total Frames: {total_frames}, Interval: {interval}")

        # Check if the video is valid
        if fps <= 0 or total_frames <= 0:
            print(f"[ERROR] Invalid video file: {video_path}. FPS: {fps}, Total Frames: {total_frames}")
            video.release()
            return frame_list

        # Calculate the range of the middle 5 seconds of the video
        start_frame = max(0, int((total_frames // 2) - (fps * 2)))  # Start frame of the middle 5 seconds
        end_frame = min(total_frames, int((total_frames // 2) + (fps * 3)))  # End frame of the middle 5 seconds

        # Debug log: Start and end frames
        print(f"[DEBUG] Video: {video_path}, Start Frame: {start_frame}, End Frame: {end_frame}")

        frame_idx = 0
        orig_scale = None
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Debug log: Current frame index and success status
            #print(f"[DEBUG] Frame Index: {frame_idx}, Success: {success}")

            # Only process frames within the range of the middle 5 seconds
            if start_frame <= frame_idx < end_frame:
                remainder = (frame_idx - start_frame) % interval
                #print(f"[DEBUG] Frame Index: {frame_idx}, Remainder: {remainder}, Interval: {interval}")

                # Use floating-point approximation to resolve precision issues
                if abs(remainder) < 1e-6:  # Close to 0
                    orig_h, orig_w = frame.shape[:2]
                    if orig_scale is None:
                        orig_scale = min(orig_h, orig_w)

                    if self.downsample_ratio != 1.0:
                        new_size = (int(orig_w * self.downsample_ratio),
                                    int(orig_h * self.downsample_ratio))
                        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                    frame = frame[None].to(self.device)
                    frame_list.append(frame)

                    # Debug log: Frame added to the list
                    #print(f"[DEBUG] Frame added at index: {frame_idx}")

            frame_idx += 1

            # Stop processing if the end frame is reached
            if frame_idx >= end_frame:
                break

        video.release()

        # Log a warning if no frames were extracted
        if not frame_list:
            print(f"[WARNING] No frames were extracted from the video: {video_path}")
        else:
            print(f"[INFO] Extracted {len(frame_list)} frames from the video: {video_path}")

        return frame_list, orig_scale
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        orig_scale = None
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            orig_h, orig_w = frame.shape[:2]
            if orig_scale is None:
                orig_scale = min(orig_h, orig_w)

            if self.downsample_ratio != 1.0:
                new_size = (int(orig_w * self.downsample_ratio),
                            int(orig_h * self.downsample_ratio))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)

        if not frame_list:
            raise RuntimeError(f"No images found in {img_folder}")
        return frame_list, orig_scale    



def dynamic_degree(dynamic, video_list):
    """
    Calculate the dynamic degree for a list of videos.

    Args:
        dynamic (DynamicDegree): The DynamicDegree object for processing videos.
        video_list (list): List of video paths to process.

    Returns:
        tuple: Average score across all videos and individual video results.
    """
    sim = []  # List to store average scores for each video
    video_results = []  # List to store results for each video

    for video_path in tqdm(video_list, disable=get_rank() > 0):
        # Process each video and calculate scores
        try:
            whether_move, total_score, avg_score = dynamic.infer(video_path)
        except Exception as e:
            print(f"[DD][ERROR] {video_path} -> {e}")
            whether_move, total_score, avg_score = None, None, None

        # optional︰留下 debug 訊息；若三者都是 None 就不印
        if avg_score is not None:
            print(f"[INFO] {video_path}  move={whether_move}  "
                  f"total={total_score:.4f}  avg={avg_score:.4f}")

        # Append the result for the current video
        video_results.append(
            {"video_path": video_path,
             "video_results": -1 if avg_score is None else float(avg_score)}
        )

        if avg_score is not None and not np.isnan(avg_score):
            sim.append(avg_score)

    # Calculate the overall average score across all videos
    avg_score = np.mean(sim) if sim else -1
    return avg_score, video_results



def compute_dynamic_degree(json_dir, device, submodules_list, **kwargs):
    """
    Compute the dynamic degree for a list of videos.

    Args:
        json_dir (str): Path to the JSON directory containing video information.
        device (torch.device): The device to run the model on.
        submodules_list (dict): Dictionary containing model paths and configurations.

    Returns:
        tuple: Overall average score and detailed video results.
    """
    model_path = submodules_list["model"]
    # Set arguments for the RAFT model
    args_new = edict({"model": model_path, "small": False, "mixed_precision": False, "alternate_corr": False})
    dynamic = DynamicDegree(args_new, device, downsample_ratio=0.5)

    # Load video list and distribute it across ranks
    video_list, _ = load_dimension_info(json_dir, dimension='dynamic_degree', lang='en')
    video_list = distribute_list_to_rank(video_list)

    # Calculate dynamic degree for all videos
    all_results, video_results = dynamic_degree(dynamic, video_list)

    # Handle distributed results if running on multiple GPUs
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)

    return all_results, video_results
