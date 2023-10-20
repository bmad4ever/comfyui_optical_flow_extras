## About

This package complements (**and requires**) the [comfyui-optical-flow](https://github.com/seanlynch/comfyui-optical-flow) package.

## Nodes


| Node            | Description                                                                                                                                                                                                                                                                                               |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GaussianBlur    | Applies gaussian blur in the x and y components of the optical flow using a kernel of size: (`kernel_size+1`,`kernel_size+1`).
| ScaleMagnitudes | The optical flow magnitudes are multiplied by the value of `scale`.
| ClampMagnitudes    | Sets to `clamp_max_magnitude` all the magnitudes above it. Potentially useful to clear outliers. 
| LowerBoundThresholding    | Sets to zero all the magnitudes below `threshold`.
| MaskOpticalFlow    | Clears** the optical flow outside of the roi defined by the mask. Potentially useful when there is a well-defined region of interest. <br><br>**the flow magnitudes are multiplied with the mask. 
| OpticalFlowToMask    | Creates a binary mask from the optical flow magnitudes. <br><br>The optical flow can be applied to the mask by setting `method` to `adjusted`. <br>Setting it to `combined` will combine the first obtained mask with the adjusted mask.
