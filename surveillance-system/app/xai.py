"""
Explainability helpers.

Primary XAI method: rule trace + metadata summary (works everywhere).
Optional: Grad-CAM heatmap using Captum, if torch model available.
"""
import os
import cv2
import logging

logger = logging.getLogger(__name__)


def rule_trace_explanation(event):
    """Human-readable explanation text (already provided by behavior module)."""
    return event.get("explanation", "No explanation available")


def structured_explanation(event):
    """Return a dict structure suitable for frontend rendering.

    IMPORTANT: We must not embed a reference to event['metadata'] itself,
    or json.dumps() will detect a circular reference.
    """
    # Make a shallow copy of metadata and remove any 'xai' key
    # so we don't end up embedding ourselves inside ourselves.
    features = dict(event.get("metadata", {}))
    features.pop("xai", None)

    return {
        "event_type": event["type"],
        "person": event["person"],
        "severity": event["severity"],
        "confidence": event["confidence"],
        "text": event["explanation"],
        "features": features,
    }


def try_gradcam(frame, model, target_layer, input_tensor_fn):
    """
    Optional Grad-CAM using Captum. Returns heatmap overlaid image or None.
    Requires a torch model. Safe to skip if not configured.
    """
    try:
        from captum.attr import LayerGradCam
        import torch
        import numpy as np
        layer_gc = LayerGradCam(model, target_layer)
        attributions = layer_gc.attribute(input_tensor_fn(), target=0)
        heatmap = attributions.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        return overlay
    except Exception as e:
        logger.warning(f"Grad-CAM skipped: {e}")
        return None


def save_explanation_image(frame, event, out_dir):
    """Draw explanation text on the frame and save as image."""
    img = frame.copy()
    text_lines = [
        f"EVENT: {event['type']}",
        f"PERSON: {event['person']}",
        f"CONF: {event['confidence']:.2f}",
        f"EXPL: {event['explanation'][:60]}",
    ]
    y = 30
    for line in text_lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        y += 25
    path = os.path.join(out_dir, f"xai_{event['type']}_{int(__import__('time').time())}.jpg")
    cv2.imwrite(path, img)
    return path