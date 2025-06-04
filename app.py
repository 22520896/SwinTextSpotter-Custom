import os
import time
import gradio as gr
from PIL import Image
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy
from detectron2.utils.logger import setup_logger
from projects.SWINTS.swints import add_SWINTS_config
from demo.predictor import VisualizationDemo
from detectron2.utils.visualizer_vintext import decoder

# Setup logger
setup_logger()

# Configure Detectron2
def setup_cfg():
    cfg = get_cfg()
    add_SWINTS_config(cfg)
    cfg.merge_from_file('./projects/SWINTS/configs/SWINTS-swin-finetune-vintext.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS', './output/model_vintext.pth'])
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.4
    cfg.freeze()
    return cfg

cfg = setup_cfg()
demo = VisualizationDemo(cfg)


# Character mapping list (CTLABELS)
CTLABELS = [
    " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_",
    "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ˋ",
    "ˊ", "﹒", "ˀ", "˜", "ˇ", "ˆ", "˒", "‑"
]



def _ctc_decode_recognition(rec):
    last_char = False
    s = ''
    for c in rec:
        c = int(c)
        if 0 < c < 107:
            s += CTLABELS[c-1]
            last_char = c
        elif c == 0:
            s += u''
        else:
            last_char = False
    if len(s) == 0:
        s = ' '
    s = decoder(s)
    return s

# Format prediction results for display
def format_predictions(predictions, image_shape, filtered_best_candidates):
    instances = predictions["instances"]
    num_instances = len(instances) if hasattr(instances, '__len__') else instances.pred_boxes.tensor.shape[0]
    image_height, image_width = image_shape[:2]
    pred_boxes = instances.pred_boxes.tensor
    scores = instances.scores
    pred_rec = instances.pred_rec  # Raw tensor from the model

    # Decode each instance using _ctc_decode_recognition
    pred_texts = [_ctc_decode_recognition(rec) for rec in pred_rec]

    # Format the output with better structure and readability
    output = "═══════════════════════════════════════════════════════\n"
    output += f"📏 Image Shape: {image_height} x {image_width}\n"
    output += f"📌 Number of Instances: {num_instances}\n"
    output += "═══════════════════════════════════════════════════════\n\n"

    for i in range(num_instances):
        box = pred_boxes[i].tolist()
        score = scores[i].item()
        text = pred_texts[i]
        candidate = filtered_best_candidates[i] if i < len(filtered_best_candidates) else "N/A"
        output += f"Instance #{i+1}:\n"
        output += f"  🟦 Bounding Box: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]\n"
        output += f"  📊 Confidence Score: {score:.4f}\n"
        output += f"  ✍️ Recognized Text: {text} → {candidate}\n"
        output += "-------------------------------------------------------\n\n"
    
    return output

# Predict and process the image
def predict_with_gradio(image):
    if image is None:
        return None, None, "Please upload an image before analyzing!"
    
    # Convert PIL image to NumPy array in BGR format directly
    img = convert_PIL_to_numpy(image, format="BGR")

    # Run prediction
    predictions, visualized_output, vis_output_dict, filtered_best_candidates = demo.run_on_image(img, 0.4, None)

    # Convert visualized output to PIL image without saving
    output_np_pred = visualized_output.get_image()  # Shape: (H, W, 3), RGB
    output_image_pred = Image.fromarray(output_np_pred)

    output_np_dict = vis_output_dict.get_image()  # Shape: (H, W, 3), RGB
    output_image_dict = Image.fromarray(output_np_dict)

    formatted_output = format_predictions(predictions, img.shape, filtered_best_candidates)

    return output_image_pred, output_image_dict, formatted_output

# Gradio interface
with gr.Blocks(title="SwinTextSpotter Gradio Demo") as demo_gr:
    gr.Markdown(
        """
        # 📸 Vietnamese Scene Text Spotting
        """
    )

    # Row for input image at the top
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📤 Input Image")

    # Row for centered button
    with gr.Row():
        with gr.Column(scale=1, min_width=200, elem_id="centered-button"):
            submit_btn = gr.Button("🚀 Analyze Image", variant="primary")

    # Row for two output images below
    with gr.Row():
        with gr.Column(scale=1):
            output_image_pred = gr.Image(type="pil", label="📥 Recognized Text")
        with gr.Column(scale=1):
            output_image_dict = gr.Image(type="pil", label="📥 Candidate Text")


    # Full-width Textbox
    with gr.Row():
        output_text = gr.Textbox(label="📄 Prediction Results", lines=15, max_lines=100, show_copy_button=True)

    submit_btn.click(fn=predict_with_gradio, inputs=image_input, outputs=[output_image_pred, output_image_dict, output_text])

demo_gr.launch(share=True)