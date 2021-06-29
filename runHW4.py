import train_w_o_adn
from dataset import prepare_data
import easydict

dPath = "/media/kayrish52/DataStorage/NWPU"

prepare_data(data_path=dPath, patch_size=40, stride=10, aug_times=1)

savePath = "4BlkNoNoise"
opt = easydict.EasyDict({
       "preprocess": False,
       "batchSize": 1152,
       "num_of_layers": 4,
       "epochs": 80,
       "milestone": 30,
       "lr": 1e-3,
       "outf": "logs",
       "mode": "B",
       "noiseL": 20.0,
       "val_noiseL": 20.0
})
noiseMethod = 0
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "8BlkNoNoise"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 896,
        "num_of_layers": 8,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 0
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "4BlkAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 1152,
        "num_of_layers": 4,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 1
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "8BlkAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 896,
        "num_of_layers": 8,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 1
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "4BlkMotionPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 1152,
        "num_of_layers": 4,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 4
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "8BlkMotionPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 896,
        "num_of_layers": 8,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 4
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "4BlkSmoothingPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 1152,
        "num_of_layers": 4,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 5
train_w_o_adn.main(opt, savePath, noiseMethod)

savePath = "8BlkSmoothingPlusAdditive"
opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 896,
        "num_of_layers": 8,
        "epochs": 80,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})
noiseMethod = 5
train_w_o_adn.main(opt, savePath, noiseMethod)

