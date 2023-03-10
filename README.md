## Paint-With-Words

[Paper](https://arxiv.org/abs/2211.01324) | [Demo](https://huggingface.co/spaces/nyanko7/sd-diffusers-webui)

Paint-with-words is a method proposed by researchers from NVIDIA that allows users to control the location of objects by selecting phrases and painting them on the canvas. The user-specified masks increase the value of corresponding entries of the attention matrix in the cross-attention layers. 

Inspired by this method, we created a simple a1111-style sketching UI that allows multi-mask input to address same area on different tokens. Also, textual-inversion and LoRA support are fully functional*, you can add them to the generation process and adjust the strength and area they are applied to.

**Config and Run**

1. Set your model path in https://github.com/Mikubill/sd-paint-with-words/blob/15e800e6c5ec14763567ec47173c9528fedc2649/app.py#L28-L35
2. `python app.py`

**Some samples**

| Sketch | Image |
|:-------------------------:|:-------------------------:|
|<img width="512" alt="" src="https://github.com/Mikubill/sd-paint-with-words/blob/main/samples/sample-3-1.png?raw=true">  |  <img width="512" alt="" src="https://github.com/Mikubill/sd-paint-with-words/blob/main/samples/sample-3-2.png?raw=true"> |
|<img width="512" alt="" src="https://github.com/Mikubill/sd-paint-with-words/blob/main/samples/sample-1-compressed.png?raw=true">  |  <img width="512d" alt="" src="https://github.com/Mikubill/sd-paint-with-words/blob/main/samples/sample-1-output-compressed.png?raw=true"> |
