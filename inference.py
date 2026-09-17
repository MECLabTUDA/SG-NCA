from vid_inference.inference_runner import InferenceRunner
from training_utils.trainer import BaseTrainer
import imageio



if False:
    trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dazzling-carnation-75") # NCA")
    video_reader = imageio.get_reader("/local/scratch/Cholec80/cholec80_full_set/videos/video12.mp4") # <- val sample video
    out_folder = "/local/scratch/clmn1/videoNCA/inference/Cholec/12"
else:
    trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/confused-flower-107") # NCA")
    trainer.relation_config["feature_extraction"]["area_threshold"] = 1500
    video_reader = imageio.get_reader("/local/scratch/Catarakt/videos/micro/train05.mp4") # <- val sample video
    out_folder = "/local/scratch/clmn1/videoNCA/inference/Catarakt/05"


inference_runner = InferenceRunner(trainer, video_reader, out_folder)
inference_runner.threshold = 0.4
#inference_runner.top_k_sampling = True

inference_runner.run_inference(50_000)