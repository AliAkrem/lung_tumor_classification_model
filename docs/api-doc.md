
copy
$ npm i -D @gradio/client
2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. If this is a private Space, you may need to pass your Hugging Face token as well (read more). Or use the
API Recorder

 to automatically generate your API requests.

API name: /predict_tumor Main prediction function for Gradio interface
copy
import { Client } from "@gradio/client";

const response_0 = await fetch("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png");
const exampleImage = await response_0.blob();

const client = await Client.connect("aliakrem/lung_tumor_prediction");
const result = await client.predict("/predict_tumor", {
				image: exampleImage,
});

console.log(result.data);
Accepts 1 parameter:
image Blob | File | Buffer Required

The input value that is provided in the "Upload Lung CT Scan Image" Image component. For input, either path or url must be provided. For output, path is always provided.

Returns 1 element
string

The output value that appears in the "Prediction Results" Markdown component.

API name: /predict_tumor_json API endpoint that returns JSON results
copy
import { Client } from "@gradio/client";

const response_0 = await fetch("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png");
const exampleImage = await response_0.blob();

const client = await Client.connect("aliakrem/lung_tumor_prediction");
const result = await client.predict("/predict_tumor_json", {
				image: exampleImage,
});

console.log(result.data);
Accepts 1 parameter:
image Blob | File | Buffer Required

The input value that is provided in the "Upload Image for API Test" Image component. For input, either path or url must be provided. For output, path is always provided.

Returns 1 element

The output value that appears in the "API Response" Json component.
