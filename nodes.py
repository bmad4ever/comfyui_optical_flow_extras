import numpy as np
import cv2
import importlib
optical_flow_nodes = importlib.import_module("custom_nodes.comfyui-optical-flow.optical_flow")


class GaussianBlurOpticalFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW", ),
            "kernel_size": ("INT", {"default": 4, "min": 2, "step": 2}),
        }}

    RETURN_TYPES = ("OPTICAL_FLOW", )
    FUNCTION = "blur_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def blur_1(optical_flow, kernel_size):
        flow_x = optical_flow[..., 0]
        flow_y = optical_flow[..., 1]

        # ??? maybe should allow more control over the kernels?
        blurred_flow_x = cv2.GaussianBlur(flow_x, (kernel_size + 1, kernel_size + 1), 0)
        blurred_flow_y = cv2.GaussianBlur(flow_y, (kernel_size + 1, kernel_size + 1), 0)

        new_optical_flow = np.empty_like(optical_flow)
        new_optical_flow[..., 0] = blurred_flow_x
        new_optical_flow[..., 1] = blurred_flow_y

        return new_optical_flow

    def blur_all(self, optical_flow, kernel_size):
        return ([self.blur_1(f, kernel_size) for f in optical_flow ], )


class ScaleOpticalFlowMagnitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW",),
            "scale": ("FLOAT", {"default": 2, "min": 0, "step": .01}),
        }}

    RETURN_TYPES = ("OPTICAL_FLOW",)
    FUNCTION = "scale_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def scale_1(optical_flow, scale):
        new_optical_flow = optical_flow * scale
        return new_optical_flow

    def scale_all(self, optical_flow, scale):
        return ([self.scale_1(f, scale) for f in optical_flow],)


class ClampOpticalFlowMagnitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW", ),
            "clamp_max_magnitude": ("FLOAT", {"default": 1, "min": 0, "step": .01}),
        }}

    RETURN_TYPES = ("OPTICAL_FLOW", )
    FUNCTION = "clamp_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def clamp_1(optical_flow, clamp_max_magnitude):
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        mag[mag > clamp_max_magnitude] = clamp_max_magnitude
        xs, ys = cv2.polarToCart(mag, ang)
        new_optical_flow = np.empty_like(optical_flow)
        new_optical_flow[..., 0] = xs
        new_optical_flow[..., 1] = ys
        return new_optical_flow

    def clamp_all(self, optical_flow, clamp_max_magnitude):
        return ([self.clamp_1(f, clamp_max_magnitude) for f in optical_flow],)


class LowerBoundThresholdingOpticalFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW", ),
            "threshold": ("FLOAT", {"default": .1, "min": 0, "step": 0.001})
        }}

    RETURN_TYPES = ("OPTICAL_FLOW", )
    FUNCTION = "zero_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def zero_1(optical_flow, threshold):
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        mag[mag <= threshold] = 0
        xs, ys = cv2.polarToCart(mag, ang)
        new_optical_flow = np.empty_like(optical_flow)
        new_optical_flow[..., 0] = xs
        new_optical_flow[..., 1] = ys
        return new_optical_flow

    def zero_all(self, optical_flow, threshold):
        return ([self.zero_1(f, threshold) for f in optical_flow],)


class OpticalFlowToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW", ),
            "magnitude_threshold": ("FLOAT", {"default": .1, "min": 0, "step": 0.001}),
            "method": ([
                "simple",    # just consider the magnitudes
                "adjusted",  # apply the optical flow to the mask
                "combined",  # combine adjusted and normal
            ],)
        }}

    RETURN_TYPES = ("MASK", )
    FUNCTION = "to_binmask_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def to_binmask_1(optical_flow, magnitude_threshold, method):
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        mag[mag <= magnitude_threshold] = 0
        mag[mag > magnitude_threshold] = 1

        if method != "simple":
            adjusted = optical_flow_nodes.image_transform_optical_flow(mag, optical_flow)

            # note: except for 0s, all other values should become 255

            if method == "adjusted":
                adjusted[adjusted > 0] = 1
                return adjusted*255

            # if method == combined:
            combined = mag + adjusted
            combined[combined > 0] = 1
            return combined*255

        # if method == normal:
        return mag*255

    def to_binmask_all(self, optical_flow, magnitude_threshold, method):
        masks = [self.to_binmask_1(f, magnitude_threshold, method) for f in optical_flow]
        masks = optical_flow_nodes.np2tensor(masks)
        return (masks, )


class MaskOpticalFlow:
    def __init__(self):
        self.mask_index = 0

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "optical_flow": ("OPTICAL_FLOW", ),
            "mask": ("MASK", ),
        }}

    RETURN_TYPES = ("OPTICAL_FLOW", )
    FUNCTION = "mask_all"
    CATEGORY = "Optical flow/extras"

    @staticmethod
    def mask_1(optical_flow, mask):
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        mag = mag * mask.detach().cpu().numpy()
        xs, ys = cv2.polarToCart(mag, ang)
        new_optical_flow = np.empty_like(optical_flow)
        new_optical_flow[..., 0] = xs
        new_optical_flow[..., 1] = ys
        return new_optical_flow

    def mask_all(self, optical_flow, mask):
        def get_next_mask():
            self.mask_index = min(self.mask_index + 1, mask.shape[0] - 1)
            return mask[self.mask_index]

        return ([self.mask_1(f, get_next_mask()) for f in optical_flow], )


NODE_CLASS_MAPPINGS = {
    "GaussianBlur (flow)": GaussianBlurOpticalFlow,
    "ScaleMagnitudes (flow)": ScaleOpticalFlowMagnitude,
    "ClampMagnitudes (flow)": ClampOpticalFlowMagnitude,
    "LowerBoundThresholding (flow)": LowerBoundThresholdingOpticalFlow,
    "OpticalFlowToMask": OpticalFlowToMask,
    "MaskOpticalFlow": MaskOpticalFlow
}


