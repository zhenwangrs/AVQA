import open_clip
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer
from open_clip import tokenizer

url = "http://images.cocodataset.org/val2017/000000039769.jpg"


model, _, preprocess = open_clip.create_model_and_transforms('coca_ViT-B-32', pretrained='laion2b_s13b_b90k')
state_dict = model.state_dict()
# tokenizer = open_clip.get_tokenizer('coca_ViT-B-32-laion2B-s13B-b90k')
# clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text = tokenizer.tokenize(["a diagram", "a dog", "a big cat"])
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0)
# text2 = clip_tokenizer(["a diagram", "a dog", "a big cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features, text_embeds = model._encode_text(text)
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]