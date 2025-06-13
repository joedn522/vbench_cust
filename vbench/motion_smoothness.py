import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from vbench.utils import load_dimension_info

from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


class FrameProcess:
    def __init__(self):
        pass


    def get_frames(self, video_path, target_fps=6):
        """
        Extract frames from the middle 5 seconds of a video file with a dynamically calculated interval.

        Args:
            video_path (str): Path to the video file.
            target_fps (int): Target number of frames to extract per second. Default is 6.

        Returns:
            list: List of frames in RGB format.
        """
        frame_list = []
        video = cv2.VideoCapture(video_path)

        # Get the FPS and total number of frames of the video
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Dynamically calculate frame_interval
        frame_interval = max(1, round(fps / target_fps))
        print(f"[INFO] Video FPS: {fps}, Target FPS: {target_fps}, Frame Interval: {frame_interval}")

        # Calculate the range of the middle 5 seconds of the video
        start_frame = max(0, (total_frames // 2) - int(fps * 2.5))  # Start frame of the middle 5 seconds
        end_frame = min(total_frames, (total_frames // 2) + int(fps * 2.5))  # End frame of the middle 5 seconds

        frame_idx = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Only process frames within the range of the middle 5 seconds
            if start_frame <= frame_idx < end_frame:
                if (frame_idx - start_frame) % frame_interval == 0:  # Extract frames based on the frame_interval
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                    frame_list.append(frame)

            frame_idx += 1

            # Stop processing if the end frame is reached
            if frame_idx >= end_frame:
                break

        video.release()
        print(f'Loading [video] from [{video_path}], the number of frames = [{len(frame_list)}]')
        if not frame_list:
            print(f"[WARNING] No frames were extracted from the video: {video_path}")
        return frame_list
    

    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list


    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract


class MotionSmoothness:
    def __init__(self, config, ckpt, device, downsample_ratio=1.0):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.downsample_ratio = downsample_ratio
        self.initialization()
        self.load_model()

    
    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()


    def initialization(self):
        if isinstance(self.device, torch.device) and self.device.type == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()


    def motion_score(self, video_path):
        iters = int(self.niters)
        # get inputs
        if video_path.lower().endswith(('.mp4', '.mov')):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError
        frame_list = self.fp.extract_frame(frames, start_from=0)

        inputs = []
        for frame in frame_list:
            if self.downsample_ratio != 1.0:
                new_size = (int(frame.shape[1] * self.downsample_ratio),
                            int(frame.shape[0] * self.downsample_ratio))
                frame_ds = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            else:
                frame_ds = frame
            inputs.append(img2tensor(frame_ds).to(self.device))

        if len(inputs) <= 1:
            raise RuntimeError("Too few frames")
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # -----------------------  Interpolater ----------------------- 
        # print(f'Start frame interpolation:')
        for i in range(iters):
            # print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        # -----------------------  cal_vfi_score ----------------------- 
        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]

        if self.downsample_ratio != 1.0:
            new_size = (int(frames[0].shape[1] * self.downsample_ratio),
                        int(frames[0].shape[0] * self.downsample_ratio))
            frames_ds = [cv2.resize(f, new_size, interpolation=cv2.INTER_LINEAR)
                         for f in frames]
            vfi_score = self.vfi_score(frames_ds, outputs)
        else:
            vfi_score = self.vfi_score(frames, outputs)
        norm = (255.0 - vfi_score)/255.0
        return norm


    def vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self.get_diff(ori[i], interpolate[i]))
        return np.mean(np.array(scores))


    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)



def motion_smoothness(motion, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        try:
            score = motion.motion_score(video_path)
        except Exception as e:
            # 單影片失敗：記 -1、寫 log、繼續跑
            print(f"[MS][ERROR] {video_path} -> {e}")
            score = None

        if score is None or np.isnan(score):
            video_results.append(
                {"video_path": video_path, "video_results": -1}
            )
        else:
            video_results.append(
                {"video_path": video_path, "video_results": float(score)}
            )
            sim.append(score)
    
    avg_score = np.mean(sim) if sim else -1
    return avg_score, video_results


def compute_motion_smoothness(json_dir, device, submodules_list, **kwargs):
    config = submodules_list["config"] # pretrained/amt_model/AMT-S.yaml
    ckpt = submodules_list["ckpt"] # pretrained/amt_model/amt-s.pth
    motion = MotionSmoothness(config, ckpt, device, downsample_ratio=0.5) 
    video_list, _ = load_dimension_info(json_dir, dimension='motion_smoothness', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = motion_smoothness(motion, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
