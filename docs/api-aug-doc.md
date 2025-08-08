API name: /create_individual_augmentations Create each augmentation as a separate image.
copy
import { Client } from "@gradio/client";

const response_0 = await fetch("https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png");
const exampleImage = await response_0.blob();

const client = await Client.connect("aliakrem/aug");
const result = await client.predict("/create_individual_augmentations", {
				image: exampleImage,
});

console.log(result.data);
Accepts 1 parameter:
image Blob | File | Buffer Required

The input value that is provided in the "Upload Image" Image component. For input, either path or url must be provided. For output, path is always provided.

Returns list of 8 elements
[0] string

The output value that appears in the "Original" Image component.

[1] string

The output value that appears in the "Random Rotation" Image component.

[2] string

The output value that appears in the "Width Shift" Image component.

[3] string

The output value that appears in the "Height Shift" Image component.

[4] string

The output value that appears in the "Shear" Image component.

[5] string

The output value that appears in the "Zoom" Image component.

[6] string

The output value that appears in the "Brightness" Image component.

[7] string

The output value that appears in the "Channel Shift" Image component.
