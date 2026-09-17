from training_utils.trainer import IncrementalNCATrainer
from data.cholec import CholecDataset
from data.cataracts import CataractsDataset


def train_cholec():
    dryrun = False
    trainer = IncrementalNCATrainer(CholecDataset)
    trainer.incremental_nca_params["num_channels"] = 4
    trainer.incremental_nca_params["hidden_size"] = 8
    trainer.train([2,4,5,6,9], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([1,3,10], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([7], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([8], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")

def train_cataracts():
    dryrun = False
    trainer = IncrementalNCATrainer(CataractsDataset)
    trainer.incremental_nca_params["num_channels"] = 8
    trainer.incremental_nca_params["hidden_size"] = 8
    trainer.incremental_nca_params["num_epochs_per_increment"] = 20

    #trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/daily-wave-61", load_best=True)
    print(f"{trainer.incremental_nca_params['num_channels']=}, {trainer.incremental_nca_params['hidden_size']=}")

    
    trainer.train([1,2,3,5,7], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")

    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([4,6,14], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([8,12,16], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([9,10], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([15], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([13], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([11], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")
    trainer.train([17], dryrun=dryrun)
    trainer.save_checkpoint()
    trainer.eval("val", "dices")
    trainer = IncrementalNCATrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
    trainer.eval("val", "dices_best")

#IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dummy-2lzqhh0z/").eval("val")
#exit()

train_cataracts()