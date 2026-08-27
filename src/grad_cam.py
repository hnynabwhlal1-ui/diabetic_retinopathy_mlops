import tensorflow as tf
import numpy as np
import cv2
from src.config import LAST_CONV_LAYER_NAME, BASE_MODEL_NAME


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME, base_model_name=BASE_MODEL_NAME, pred_index=None):
    """
    Generate a Grad-CAM heatmap for the specified target convolutional layer,
    handling nested sub-models (like EfficientNetB1).
    """
    # 1. Access the inner sub-model and target conv layer
    try:
        sub_model = model.get_layer(base_model_name)
        target_layer = sub_model.get_layer(last_conv_layer_name)
    except Exception:
        sub_model = model
        target_layer = model.get_layer(last_conv_layer_name)

    # 2. Build feature extractor model from the sub-model
    grad_model = tf.keras.Model(
        inputs=sub_model.inputs,
        outputs=[target_layer.output, sub_model.output]
    )

    # 3. Compute gradients with GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Pass through remaining head layers if nested
        if sub_model != model:
            x = predictions
            start_index = model.layers.index(model.get_layer(base_model_name)) + 1
            for layer in model.layers[start_index:]:
                x = layer(x)
            final_predictions = x
        else:
            final_predictions = predictions

        # Handle Binary Sigmoid Output (1 Neuron) vs Categorical Output
        if final_predictions.shape[-1] == 1:
            loss = final_predictions[:, 0]
        else:
            if pred_index is None:
                pred_index = tf.argmax(final_predictions[0])
            loss = final_predictions[:, pred_index]

    # 4. Compute importance weights
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Generate raw heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU and normalization
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the Grad-CAM heatmap on top of the original fundus image.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap_colored * alpha + original_img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img