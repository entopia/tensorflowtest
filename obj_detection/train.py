import objd

"""

This script loads training and test datasets, load weights of base model 
(that is used for transfer learning) and train the model. 
Also outputs final mAP scores on training and test sets.

"""

#------------------------
# Load and prepare training and test sets
train_set = objd.PlansDataset()
train_set.load_dataset('plans', is_train=True)
train_set.prepare()
print('Train examples: %d' % len(train_set.image_ids))

test_set = objd.PlansDataset()
test_set.load_dataset('plans', is_train=False)
test_set.prepare()
print('Test examples: %d' % len(test_set.image_ids))

#------------------------
# build model and load weights (and exclude the output layers)
config = objd.PlansConfig()
config.display()

print("Loading model and weights...")
model = objd.MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
print("Loaded model and weights.")

#------------------------
# train weights (output layers or 'heads') and output final mAP scores
print("Training starting...")
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
print("Training finished")

train_mAP = objd.evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
test_mAP = objd.evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
