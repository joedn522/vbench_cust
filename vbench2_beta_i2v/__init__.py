import os

from vbench2_beta_i2v.utils import init_submodules, save_json, load_json
from vbench.utils import get_prompt_from_filename
from vbench import VBench
import importlib
from pathlib import Path
import os, csv, json


class VBenchI2V(VBench):
    def __init__(self, device, full_info_dir, output_path):
        super().__init__(device, full_info_dir, output_path)
        self.i2v_dims = ["i2v_subject", "i2v_background", "camera_motion"]
        self.quality_dims = ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "temporal_flickering", "motion_smoothness", "dynamic_degree",]
        
    def build_full_dimension_list(self, ):
        return self.i2v_dims + self.quality_dims

    def build_list_capable(self, videos_path, name, dim_list,
                        *args, **kwargs):
        suf = Path(videos_path).suffix.lower()
        if suf in {".list", ".txt", ".tsv"}:
            # 1) 讀檔 → 取第一欄 path
            with open(videos_path) as f:
                if suf == ".tsv":
                    paths = [row[0] for row in csv.reader(f, delimiter="\t") if row]
                else:
                    paths = [ln.strip().split("\t")[0] for ln in f if ln.strip()]

            cur = [{"prompt_en": get_prompt_from_filename(p),
                    "dimension": dim_list,
                    "video_list": [p]} for p in paths]

            fp = os.path.join(self.output_path, name+"_full_info.json")
            save_json(cur, fp)
            print(f"Evaluation meta data saved to {fp}")
            return fp

        return self.build_full_info_json(videos_path, name, dim_list, *args, **kwargs)

    def evaluate(self, videos_path, name, dimension_list=None, custom_image_folder=None, mode='vbench_standard', local=False, read_frame=False, resolution="1-1", **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
        # print('BEFORE BUILDING')
        cur_full_info_path = self.build_list_capable(videos_path, name, dimension_list, custom_image_folder=custom_image_folder, mode=mode)
        # print('AFTER BUILDING')
        for dimension in dimension_list:
            try:
                if dimension in self.i2v_dims:
                    dimension_module = importlib.import_module(f'vbench2_beta_i2v.{dimension}')
                else:
                    dimension_module = importlib.import_module(f'vbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')
