import objd

"""

This script loads training and test datasets, load weights of model and produces
mAP test statistics and producing plots of predictions

"""

#------------------------
# Load and prepare training and test sets
train_set = objd.PlansDataset()
train_set.load_dataset('plans', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

test_set = objd.PlansDataset()
test_set.load_dataset('plans', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

#------------------------
# build model and load weights
cfg = objd.PredictionConfig()
model = objd.MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('plans_cfg20191211T1049/mask_rcnn_plans_cfg_0001.h5', by_name=True)

#------------------------
# evaluate model using mAP on training and test sets
print("Calculating mAP metrics...")
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = objd.evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
print("Calculated mAP metrics.")

#------------------------
# Produce plots for predictions for training and test set
objd.plot_actual_vs_predicted(train_set, model, cfg)
objd.plot_actual_vs_predicted(test_set, model, cfg)
