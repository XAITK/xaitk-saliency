from typing import Dict, Any, Iterator, Sequence, Hashable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack


rand_seed = 0
np.random.seed(rand_seed)


###############################################################################
# Get MNIST data-set
print("Loading MNIST Dataset")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Limit train/test sizes for demo purposes.
train_samples = 10000
test_samples = 1000
print(f"Train samples: {train_samples}")
print(f"Test samples : {test_samples}")

print("Splitting train/test")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=train_samples,
    test_size=test_samples,
    random_state=rand_seed
)

print("Standard Scaling train/test data")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


###############################################################################
# Train a scikit-learn estimator model -- assuming as .predict_proba
print("Training model")
model = LogisticRegression(
    C=50. / train_samples,
    penalty='l1',
    solver='saga',
    tol=0.1,
    # important to make model training some sort of deterministic
    random_state=0,
)
# model = MLPClassifier(
#     hidden_layer_sizes=(50,),
#     # max_iter=10,
#     alpha=1e-4,
#     solver='sgd',
#     verbose=10,
#     random_state=1,
#     learning_rate_init=.1
# )
model.fit(X_train, y_train)
print("Training model -- Done")


###########################################
# EVERYTHING BELOW HERE IS FOR CML REPORT #
#                                         #
# Includes XAI bits.                      #
###########################################


###############################################################################
# Model score
acc = model.score(X_test, y_test)
print(acc)
with open("metrics.txt", 'w') as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")


# Confusion Matrix
print("Plotting confusion matrix")
plt.figure()
disp = plot_confusion_matrix(
    model,
    X_test,
    y_test,
    normalize='true',
    cmap=plt.cm.Blues  # noqa
)
plt.savefig('confusion_matrix.png')


###############################################################################
# Explain Model - Visual Saliency
#
# For an example image, generate a visual saliency example
#
rand_idx = np.random.randint(0, len(X_test)-1)
print(f"Generating Saliency Maps for Random test image (idx={rand_idx})")
test_image = scaler.inverse_transform(X_test[rand_idx]).reshape(28, 28)
gt_class = y_test[rand_idx]
gt_idx = model.classes_.tolist().index(gt_class)


class XaiPredictor(ClassifyImage):
    def get_labels(self) -> Sequence[Hashable]:
        return model.classes_.tolist()

    def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
        # Demo assumes input will be the correct [H,W] shape.
        test_imgs: np.ndarray = np.asarray(list(img_iter))
        test_imgs = test_imgs.reshape((-1, 28*28))
        test_imgs = scaler.transform(test_imgs)
        idx_to_class = self.get_labels()
        return (
            {idx_to_class[idx]: p_i for idx, p_i in zip(range(10), p)}
            for p in model.predict_proba(test_imgs)
        )

    def get_config(self) -> Dict[str, Any]:
        return {}


xai_predictor = XaiPredictor()

###############################################################################
# Predict class of chosen test image
r = list(xai_predictor.classify_images([test_image]))[0]
with open('test_metrics.txt', 'w') as outfile:
    pred_conf, pred_cls = max((p, c) for c, p in r.items())
    gt_conf = r[gt_class]
    outfile.write(f"* Predicted class: {pred_cls}\n")
    outfile.write(f"* Predicted conf : {pred_conf}\n")
    outfile.write(f"* GT class conf  : {gt_conf}\n")

###############################################################################
# Generate saliency heatmap for chosen test image
visual_saliency = SlidingWindowStack(
    window_size=(2, 2),
    stride=(1, 1),
    threads=4
)
sal_map = visual_saliency.generate(test_image, xai_predictor)
plt.figure()
plt.title(f"GT Class: {gt_class}")
plt.imshow(
    sal_map[gt_idx],
    cmap=plt.cm.RdBu,  # noqa
    vmin=-1, vmax=1
)
plt.colorbar()
plt.savefig(f'xai_class_single.png')


###############################################################################
# Class Aggregate Saliency
#
# Create a plot that shows the average visual saliency for a number of test
# samples.
#
# Note that the generated plot shows a somewhat "clipped" saliency range within
# [-1,1], as some models will only have *slight* regional attention if we find
# little variation in performance. Because this is an example we're not getting
# overly detailed or informative in this plot...
#

n_per_class = 5  # total of 50 test images across 10 classes.
print(f"Selecting {n_per_class} test examples per unique class")
# X_test is scaled above, lets get it back to pre-scaled form.
X_test_natural = scaler.inverse_transform(X_test).reshape(-1, 28, 28)
X_test_natural_sample = []
for cls in model.classes_:
    X_test_natural_sample.append(
        X_test_natural[np.nonzero(y_test == cls)[0][:n_per_class]]
    )
X_test_natural_sample = np.vstack(X_test_natural_sample)

print("Generating saliency maps for test samples (this could take a hot minute)")
sal_map_list = []
for t_i in X_test_natural_sample:
    sal_map_list.append(
        visual_saliency.generate(t_i, xai_predictor)
    )
global_maps = np.sum(sal_map_list, axis=0) / len(sal_map_list)

print("Plotting global average saliency maps")
plt.figure(figsize=(10, 5))
plt.suptitle("Average Heatmaps from All Images", fontsize=16)
num_classes = global_maps.shape[0]
n_cols = np.ceil(num_classes / 2).astype(int)
for i in range(num_classes):
    # Pick a constrained value-cap for this map
    gm = global_maps[i]
    vcap = max(abs(_) for _ in (gm.max(), gm.min()))

    plt.subplot(2, n_cols, i + 1)
    plt.imshow(global_maps[i], cmap=plt.cm.RdBu, vmin=-vcap, vmax=vcap)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(i)

plt.savefig("xai_perclass_agg_saliency.png")
