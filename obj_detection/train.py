from objd import *

# train set
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train examples: %d' % len(train_set.image_ids))
 
# test/val set
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test examples: %d' % len(test_set.image_ids))

# prepare config
config = KangarooConfig()
config.display()

# define the model
print("Loading model and weights...")
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
print("Loaded model and weights.")

# train weights (output layers or 'heads')
print("Training starting...")
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
print("Training finished")

train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
