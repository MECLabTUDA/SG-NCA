
from training_utils.trainer import IncrementalNCATrainer, RelationClassifierTrainer, UNetTrainer, BaseTrainer


trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/sparkling-dragon-11")
assert isinstance(trainer, UNetTrainer)
trainer.train_relation_predictor(dryrun=True)
trainer.save_checkpoint()
print(trainer.eval_relation_predictor("val"))