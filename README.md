---
title: cat-dog-classifier
app_file: gradio/app.py
sdk: gradio
sdk_version: 5.7.1
---

# Deployment with Gradio

# Introduction to Gradio  

## What is Gradio?  
Gradio is an open-source Python library that makes it incredibly easy to build user-friendly web-based interfaces for machine learning models, APIs, or any Python function. With just a few lines of code, you can create interactive UIs that allow users to upload inputs, view model outputs, and share the interface through a simple web URL.  

## Why Do We Need Gradio?  

Gradio is essential for:  

1. **Rapid Prototyping**:  
   - Gradio allows developers and data scientists to quickly create functional interfaces to test and demonstrate their models.  

2. **Collaboration**:  
   - By providing a shareable web link, Gradio makes it easy to collect feedback from teammates, stakeholders, or end-users without requiring complex setups.  

3. **Ease of Use**:  
   - It requires minimal coding and setup, making it ideal for quickly showcasing models or functions.  

4. **Iterative Development**:  
   - Testing and gathering feedback early in the development process helps improve models and ensures they meet user requirements.  

5. **Accessibility**:  
   - With no need for front-end development skills, Gradio bridges the gap between data science and end-user interaction.  


## Features of Gradio  

1. **Interactive User Interfaces**:  
   - Supports text, image, audio, video, and tabular inputs and outputs.  
   - Allows dynamic interaction with models in real-time.  

2. **Shareable Web Links**:  
   - Automatically generates a shareable URL (`share=True`) to make your app accessible online.  

3. **Customizable**:  
   - Fully customizable interfaces with various input/output components and layouts.  

4. **Supports Various Use Cases**:  
   - Ideal for machine learning demos, data exploration, and tool development.  

5. **Fast Iterations**:  
   - Lightweight framework that allows you to build and test prototypes in minutes.  


### Examples  

#### Example 1: **Hello World with Gradio**  
**Code:**  
```python  
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)  # Share results with your friends with just 1 extra parameter 🚀
```  

**Result:**  
- Shareable via a public URL using `share=True`.  

![example 1 result](output/example1.png)

---

#### Example 2: **Multiple Inputs with Gradio**  
**Code:**  
```python  
import gradio as gr

def test(name, checkbox, value):
    return f"{name=}, {checkbox=}, {value=}"

demo = gr.Interface(fn=test, inputs=[gr.Text(), gr.Checkbox(), gr.Slider(0, 100)], outputs=gr.Text())

demo.launch()
```  

**Result:**  
- Interface with three input components:  
  1. **Text**: Enter any string.  
  2. **Checkbox**: Toggle between `True` and `False`.  
  3. **Slider**: Select a value between 0 and 100.  

![example 2 result](output/example2.png)


#### Example 3: **Interacting with a Gradio App via Client**  
**Code:**  
```python  
# pip install gradio_client

from gradio_client import Client

client = Client("http://localhost:7860")
result = client.predict(
    "John",  # str  in 'name' Textbox component
    True,  # bool  in 'checkbox' Checkbox component
    # int | float (numeric value between 0 and 100) in 'value' Slider component
    0,
    api_name="/predict"
)
print(result)
```  

**Result:**  
- Demonstrates how to interact with a Gradio app programmatically using the **Gradio Client**.  
- Sends inputs to the app running at `http://localhost:7860`.  

```bash
Loaded as API: http://localhost:7860/ ✔
name='John', checkbox=True, value=0

```

### Example: **Applying Filter to Images**  

**Code:**  
```python  
import numpy as np  
import gradio as gr  

def sepia(input_img, request: gr.Request):  
    print("Request headers dictionary:", request.headers)  
    print("IP address:", request.client.host)  
    print(f"{type(input_img)=}")  
    sepia_filter = np.array([  
        [0.393, 0.769, 0.189],  
        [0.349, 0.686, 0.168],  
        [0.272, 0.534, 0.131]  
    ])  
    sepia_img = input_img.dot(sepia_filter.T)  
    sepia_img /= sepia_img.max()  
    return sepia_img  

demo = gr.Interface(  
    fn=sepia,  
    inputs=gr.Image(height=300, width=300),  
    outputs="image"  
)  
demo.launch(share=True)  
```  

**Result:**  
- **Input:** Upload an image (300x300 pixels or resized automatically).  
- **Output:** The uploaded image is transformed with a sepia filter applied.  

![example 4 result](output/example4.png)

### Example 5: **Image Classification**  

**Code:**  
```python  
import gradio as gr  
import torch  
import timm  
from PIL import Image  
import requests  

class ImageClassifier:  
    def __init__(self):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        # Create model and move to appropriate device  
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True)  
        self.model = self.model.to(self.device)  
        self.model.eval()  

        # Get model-specific transforms  
        data_config = timm.data.resolve_model_data_config(self.model)  
        self.transform = timm.data.create_transform(**data_config, is_training=False)  

        # Load ImageNet labels  
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'  
        self.labels = requests.get(url).text.strip().split('\n')  

    @torch.no_grad()  
    def predict(self, image):  
        if image is None:  
            return None  
        
        # Preprocess image  
        img = Image.fromarray(image).convert('RGB')  
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  
        
        # Get prediction  
        output = self.model(img_tensor)  
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  
        
        # Get top 5 predictions  
        top5_prob, top5_catid = torch.topk(probabilities, 5)  
        
        return {  
            self.labels[idx.item()]: float(prob)  
            for prob, idx in zip(top5_prob, top5_catid)  
        }  

# Create classifier instance  
classifier = ImageClassifier()  

# Create Gradio interface  
demo = gr.Interface(  
    fn=classifier.predict,  
    inputs=gr.Image(),  
    outputs=gr.Label(num_top_classes=5),  
    title="Basic Image Classification with Mamba",  
    description="Upload an image to classify it using the mambaout_base.in1k model",  
    examples=[  
        ["examples/cat.jpg"],  
        ["examples/dog.jpg"]  
    ]  
)  

if __name__ == "__main__":  
    demo.launch()  
```  

**Result:**  
- **Input:** Upload an image (e.g., a cat or dog photo).  
- **Output:** The model predicts the top 5 labels with their respective probabilities based on ImageNet classes.  

Output displays the top 5 predictions with probabilities as shown in below example:  

![example 5 result](output/example5.png)

### Example 6: **Batch Image Classification**  

This example demonstrates how to perform **batch image classification** using the `mambaout_base.in1k` model. It supports multiple concurrent image preprocessing, inference, and result generation.  

#### **Server**  

```python  
import gradio as gr  
import torch  
import timm  
from PIL import Image  
import numpy as np  

class ImageClassifier:  
    def __init__(self):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model = timm.create_model('mambaout_base.in1k', pretrained=True)  
        self.model = self.model.to(self.device)  
        self.model.eval()  
        
        # Set up data transforms and labels  
        data_config = timm.data.resolve_model_data_config(self.model)  
        self.transform = timm.data.create_transform(**data_config, is_training=False)  
        
        import requests  
        url = 'https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'  
        self.labels = requests.get(url).text.strip().split('\n')  
    
    @torch.no_grad()  
    def predict_batch(self, image_list, progress=gr.Progress(track_tqdm=True)):  
        if isinstance(image_list, tuple) and len(image_list) == 1:  
            image_list = [image_list[0]]  
            
        if not image_list or image_list[0] is None:  
            return [[{"none": 1.0}]]  
            
        progress(0.1, desc="Starting preprocessing...")  
        tensors = []  
        
        # Process each image in the batch  
        for image in image_list:  
            if image is None:  
                continue  
            img = Image.fromarray(image).convert('RGB')  
            tensor = self.transform(img)  
            tensors.append(tensor)  
            
        if not tensors:  
            return [[{"none": 1.0}]]  
            
        progress(0.4, desc="Batching tensors...")  
        batch = torch.stack(tensors).to(self.device)  
        
        progress(0.6, desc="Running inference...")  
        outputs = self.model(batch)  
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  
        
        progress(0.8, desc="Processing results...")  
        batch_results = []  
        for probs in probabilities:  
            top5_prob, top5_catid = torch.topk(probs, 5)  
            result = {  
                self.labels[idx.item()]: float(prob)  
                for prob, idx in zip(top5_prob, top5_catid)  
            }  
            batch_results.append(result)  
        
        progress(1.0, desc="Done!")  
        return [batch_results]  

# Create classifier instance  
classifier = ImageClassifier()  

# Create Gradio interface  
demo = gr.Interface(  
    fn=classifier.predict_batch,  
    inputs=gr.Image(),  
    outputs=gr.Label(num_top_classes=5),  
    title="Advanced Image Classification with Mamba",  
    description="Upload images for batch classification with the mambaout_base.in1k model",  
    batch=True,  
    max_batch_size=4  
)  

if __name__ == "__main__":  
    demo.launch()  
```  

#### **Client**  

```python  
from gradio_client import Client, handle_file  
import concurrent.futures  
import time  

def make_prediction(client, image_url):  
    """Make a single prediction"""  
    try:  
        result = client.predict(  
            image_list=handle_file(image_url),  
            api_name="/predict"  
        )  
        return result  
    except Exception as e:  
        return f"Error: {str(e)}"  

def main():  
    # Test image URL  
    image_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"  
    
    # Initialize client  
    client = Client("http://127.0.0.1:7860/")  
    
    print("\nSending 16 concurrent requests with the same image...")  
    start_time = time.time()  
    
    # Send concurrent requests  
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  
        futures = [  
            executor.submit(make_prediction, client, image_url)  
            for _ in range(16)  
        ]  
        
        results = []  
        for i, future in enumerate(concurrent.futures.as_completed(futures)):  
            try:  
                result = future.result()  
                results.append(result)  
                print(f"Completed prediction {i+1}/16")  
            except Exception as e:  
                print(f"Error in request {i+1}: {str(e)}")  
    
    end_time = time.time()  
    
    # Print results  
    print(f"\nAll predictions completed in {end_time - start_time:.2f} seconds")  
    print("\nResults:")  
    for i, result in enumerate(results):  
        print(f"\nRequest {i+1}:")  
        print(result)  

if __name__ == "__main__":  
    main()  
```  

---

### Result:  

Each image returns the top 5 predicted labels with probabilities from the ImageNet dataset.  


### Example Workflow  

1. **Server Side:**  
   - Launch the Gradio app.  
   - Upload a batch of images (e.g., buses, cats, dogs, etc.).  
   - View top-5 predictions for each image along with progress updates.  

2. **Client Side:**  
   - Sends multiple concurrent requests to the Gradio server using `gradio_client`.  
   - Receives predictions and logs them to the console.  

**Example Prediction (Server Output):**  

For a bus image:  
```json  
{
   "label":"minibus",
   "confidences":[
      {
         "label":"minibus",
         "confidence":0.571755051612854
      },
      {
         "label":"vacuum, vacuum_cleaner",
         "confidence":0.07250463962554932
      },
      {
         "label":"passenger_car, coach, carriage",
         "confidence":0.05805877596139908
      },
      {
         "label":"trolleybus, trolley_coach, trackless_trolley",
         "confidence":0.03005979023873806
      },
      {
         "label":"school_bus",
         "confidence":0.005981458351016045
      }
   ]
} 
```  

For more information, visit the official ![Gradio website](https://www.gradio.app/).