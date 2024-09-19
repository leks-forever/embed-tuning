

### Embed-tuning
### Raw experimental solution
Download [bible.csv](https://huggingface.co/datasets/leks-forever/bible-lezghian-russian) and place it in the [data](data) folder

Scripts:        
[neg_mining.ipynb](neg_mining.ipynb) - prepare negative samples     
[utils.py](utils.py) - split prepaired df to train/test/val     
[model_tuning.py](model_tuning.py) - finetune model

BACKLOG:
pay attention to other [models](https://github.com/avidale/encodechka)