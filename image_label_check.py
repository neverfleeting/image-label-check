import argparse
import torch
import clip
import ruclip
from PIL import Image
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument("default_image_path", nargs="?", type=str, default="train.jpeg")
args = parser.parse_args()
default_image_path = args.default_image_path

device = "cuda" if torch.cuda.is_available() else "cpu"
eng_model, eng_preprocessor = clip.load("ViT-L/14@336px", device=device)
rus_model, rus_preprocessor = ruclip.load('ruclip-vit-large-patch14-336', device=device)
predictor = ruclip.Predictor(rus_model, rus_preprocessor, device=device)

def image_label_check(lang: str, threshold: float, input_img: Image, label: str) -> str:
    """Check if cosine similarity between the image and the text encodings is greater than threshold."""
    
    if lang.startswith("eng"):
        
        input_img = eng_preprocessor(input_img).unsqueeze(0).to(device)
        label = clip.tokenize([label]).to(device)

        with torch.no_grad():
            image_features = eng_model.encode_image(input_img)
            text_features = eng_model.encode_text(label)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
               
    elif lang.startswith("rus"):
        
        with torch.no_grad():
            image_features = predictor.get_image_latents([input_img])
            text_features = predictor.get_text_latents([label])

    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarity = similarity.item()
        
    return f"{'✅' if similarity > threshold else '❌'} {similarity > threshold}; cos_sim={similarity:.2f}"

elements = {
    "rus" : {
        "markdown" : "<h1><center>Проверка соответствия изображения и описания</center></h1>",
        "dropdown" : {"value" : "rus (ruclip-vit-large-patch14-336)", "label" : "Модель",
                      "info" : "Пожалуйста, выберите модель"},
        "slider" : {"value" : 0.5, "label" : "Порог", "info" : "Пороговое значение косинусной меры"},
        "image" : {"label" : "Изображение"},
        "label" : {"label" : "Описание изображения", "value" : "кошка"},
        "output" : {"label" : "Результат", "value" : "здесь будет результат проверки"},
        "button" : {"value" : "Проверить"},
        "flag_button" : {"value" : "Отправить результат"}
    },
    "eng" : {
        "markdown" : "<h1><center>Check if a caption describes the image</center></h1>",
        "dropdown" : {"value" : "eng (ViT-L/14@336px)", "label" : "Model", "info" : "Please, choose a model"},
        "slider" : {"value" : 0.2, "label" : "Threshold", "info" : "Cosine similarity threshold"},
        "image" : {"label" : "Input image"},
        "label" : {"label" : "Caption", "value" : "cat"},
        "output" : {"label" : "Result", "value" : "the result of the check"},
        "button" : {"value" : "Check"},
        "flag_button" : {"value" : "Send results"}
    }
}

def init_ui_components(lang: dict, visible: bool = True) -> tuple[gr.Markdown, gr.Dropdown, gr.Slider, gr.Image, gr.Textbox,
                                                                  gr.Textbox, gr.Button, gr.Button]:
    """Initialize UI components."""
    
    lang = lang.get("lang", "rus")

    markdown = gr.Markdown(elements[lang]["markdown"], visible=visible)
    with gr.Row():        
        with gr.Column(scale=1):
            dropdown = gr.Dropdown(["eng (ViT-L/14@336px)", "rus (ruclip-vit-large-patch14-336)"],
                                   **elements[lang]["dropdown"], visible=visible)
            slider = gr.Slider(0, 1, **elements[lang]["slider"], visible=visible)
            image = gr.Image(value=Image.open(default_image_path), type="pil", **elements[lang]["image"], visible=visible)
            label = gr.Textbox(**elements[lang]["label"], visible=visible)
        with gr.Column(scale=1):
            output = gr.Textbox(**elements[lang]["output"], visible=visible)
            button = gr.Button(**elements[lang]["button"], visible=visible)
            flag_button = gr.Button(**elements[lang]["flag_button"], visible=visible)
            
    return markdown, dropdown, slider, image, label, output, button, flag_button

def change_ui_components(lang: str = "rus (ruclip-vit-large-patch14-336)") -> tuple[float, str, str]:
    """Change default values of the slider and textboxes according to the chosen model."""
    
    lang = lang.split()[0]
    return elements[lang]["slider"]["value"], elements[lang]["label"]["value"], elements[lang]["output"]["value"]

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
        }
    """    
logger = gr.CSVLogger()

with gr.Blocks() as demo:
    url_params = gr.JSON({}, visible=False)
    demo.load(lambda x: x, url_params, url_params, js=get_window_url_params)
    markdown, dropdown, slider, image, label, output, button, flag_button = init_ui_components(url_params.value, visible=False)
    url_params.change(init_ui_components, url_params, [markdown, dropdown, slider, image, label, output, button, flag_button])
    dropdown.change(change_ui_components, dropdown, [slider, label, output])
    logger.setup([dropdown, slider, image, label], "flagged_data_points")
    button.click(image_label_check, [dropdown, slider, image, label], output)
    flag_button.click(lambda *args: logger.flag(list(args), "flagged"),
                      [dropdown, slider, image, label, output],
                      None, preprocess=False)
 
if __name__ == "__main__":
    demo.launch(share=True)