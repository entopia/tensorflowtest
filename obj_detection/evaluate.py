import objd

# load the train dataset
train_set = objd.KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# load the test dataset
test_set = objd.KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# create config
cfg = objd.PredictionConfig()
# define the model
model = objd.MaskRCNN(mode='inference', model_dir='./', config=cfg)

# load model weights
model.load_weights('kangaroo_cfg20191211T1049/mask_rcnn_kangaroo_cfg_0001.h5', by_name=True)

# evaluate model on training dataset
print("Calculating mAP metrics...")
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = objd.evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
print("Calculated mAP metrics.")

# plot predictions for train dataset
objd.plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
objd.plot_actual_vs_predicted(test_set, model, cfg)
