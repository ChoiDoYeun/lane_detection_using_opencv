# **lane_detection_using_opencv**

## **Table of Contents**
1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [Pipeline](#2-pipeline)
   1. [Image Preprocessing](#21-image-preprocessing)
      1. [Why Choose the L Channel of HLS and Use the Prewitt Filter](#211-why-choose-the-l-channel-of-hls-and-use-the-prewitt-filter)
      2. [Convert to HLS](#212-convert-to-hls)
      3. [Edge Detection](#213-edge-detection)
      4. [Perspective Transformation](#214-perspective-transformation)
      5. [Binarization](#215-binarization)
   2. [Sliding Window](#22-sliding-window)
      1. [Calculate Histogram](#221-calculate-histogram)
      2. [Set Initial Window Positions](#222-set-initial-window-positions)
      3. [Iterate and Move Windows](#223-iterate-and-move-windows)
      4. [Visualize Windows](#224-visualize-windows)
      5. [Fit Polynomial](#225-fit-polynomial)
   3. [Draw Lane Pipeline](#23-draw-lane-pipeline)
   4. [Main Pipeline](#24-main-pipeline)
   5. [Results](#25-results)
4. [Discussion](#3-discussion)
5. [Conclusion](#4-conclusion)


## **Abstract**

The Lane Detection Project focuses on implementing a computer vision pipeline to accurately detect lane lines on the road from video footage. Utilizing techniques such as color space conversion, edge detection, perspective transformation, and polynomial fitting, the pipeline processes each frame of the video to identify and annotate lane lines. The results demonstrate a robust system capable of handling various driving conditions effectively.


## **1. Introduction**


Lane detection is a critical component of autonomous driving systems, providing essential information for vehicle navigation and control. This project presents a comprehensive approach to lane detection using OpenCV and Python. The pipeline is designed to handle various road conditions and perspectives by leveraging several image processing techniques.


The primary steps of the pipeline include:


1. **Image Processing**: Converting the image to HLS color space, extracting the L channel, and applying edge detection.
2. **Perspective Transformation**: Generating a bird's eye view of the road to simplify lane detection.
3. **Sliding Window**: Using a histogram-based sliding window technique to identify lane line pixels and fit polynomials to these pixels.
4. **Lane Drawing**: Overlaying the detected lane lines back onto the original image and annotating the vehicle's position relative to the lane center.


## **2. PipeLine**

![flow chart](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2_flowchart.png)

### **2.1 [Image Preprocessing](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/notebook/2.1_Image_Preprocessing.ipynb)**

Image preprocessing is the first step in lane detection, where various transformations are applied to the input image to facilitate lane detection. This step includes color space conversion, edge detection, perspective transformation, and binarization.

### **2.1.1 Why Choose the L Channel of HLS and Use the Prewitt Filter**

To determine which channel to use from HLS and HSV color spaces, and which operator to use for edge detection, we modified the Image Preprocessing pipeline by extracting different channels and applying various operators. We then analyzed the resulting histograms to make an informed choice.

![binary_warrped_img & histogram](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.1.1.png)

In the sliding window technique, lanes are detected based on the peak values on the left and right sides of the histogram, divided at the midpoint. Therefore, it is crucial to have clear peak values on both sides and minimal noise in the image.

Except for the HLS L channel with Sobel X and Prewitt, HSV channel 2 with Canny, and HSV channel 3 with Sobel Y, other combinations resulted in noise or significant values on only one side of the lane. Among the four conditions that met my criteria, the HLS L channel with the Prewitt operator produced the highest value for the lower peak between the left and right maxima. Thus, this method was chosen.

### **2.1.2 Convert to HLS**

Convert the image to the HLS color space to separate the hue, lightness, and saturation components. This makes it easier to extract useful information such as lightness for lane detection.

```python
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
l_channel = hls[:, :, 1] 
```

![Convert to HLS](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.1.2_Lchannel.png)

### **2.1.3 Edge Detection**

Apply the Prewitt filter to detect edges in the image. The Prewitt filter emphasizes edges in both the horizontal and vertical directions.

```python
kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_edges = cv2.filter2D(l_channel, -1, kernelx)
```

![prewitt Edge Detection](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.1.3_prewitt.png)

### **2.1.4 Perspective Transformation**

Transform the image to a **bird's eye view** to represent the road plane in an orthogonal projection. This makes it easier to detect lanes.

```python
matrix = cv2.getPerspectiveTransform(src_mask, dst_mask)
warped_img = cv2.warpPerspective(prewitt_edges, matrix, (image.shape[1], image.shape[0]))
```

![Bird Eyes View](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.1.4_birdeyesview.png)

### **2.1.5 Binarization**

Convert the image to binary (black and white) to further emphasize the edges and facilitate lane detection.

```python
_, binary = cv2.threshold(warped_img, 127, 255, cv2.THRESH_BINARY)
```

![binary_warrped_img](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.1.5_binary.png)

### **2.2 [Sliding Window](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/notebook/2.2_Sliding_Window.ipynb)**

The sliding window technique is a crucial step for detecting lane lines in a binarized image. This step involves using a histogram to set initial window positions and iteratively moving the windows to detect lane lines.

![slidingwindows_result](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.2_slidingwindows_result.png)

### **Sliding Windows Steps**

**2.2.1 Calculate Histogram**: Calculate the histogram of pixel values in the lower half of the binarized image. This helps to find the position of the lane lines.

```python
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
```

**2.2.2 Set Initial Window Positions**: Find the peak of the histogram to set the initial window positions. Set them separately for the left and right lanes.

```python
midpoint = np.int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

**2.2.3 Iterate and Move Windows**: Starting from the set initial positions, move the windows upward to find lane pixels. Set the new window center to the average position of the found pixels.

```python
for window in range(nwindows):
    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    win_y_high = binary_warped.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

**2.2.4 Visualize Windows**: Visualize each window and the detected lane pixels. This makes it easy to verify the lane detection results.

```python
out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
for window in range(nwindows):
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
```

**2.2.5 Fit Polynomial**: Fit a second-degree polynomial to the detected lane pixels.

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

### **2.3 [Draw Lane Pipeline](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/notebook/2.3_Draw%20Lane%20Pipeline.ipynb)**

The draw lane pipeline overlays the detected lanes onto the original image. This step involves calculating the lane center and vehicle offset, and annotating the image with this information.

![Add img](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.3_add_image.png)

### **Draw Lane Steps**

**2.3.1 Generate Lane Points**: Use the polynomial fit to generate points for the lane lines.

```python
plot_y = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
right_fit_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
```

**2.3.2 Visualize Lane Area**: Use the generated lane points to visualize the lane area.

```python
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
pts = np.hstack((pts_left, pts_right))

cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
```

**2.3.3 Inverse Perspective Transform**: Transform the lane area back to the original image perspective and overlay it onto the original image.

```python
newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(matrix), (image.shape[1], image.shape[0]))
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
```

**2.3.4 Calculate Lane Center and Vehicle Offset**: Calculate the lane center and vehicle offset, and annotate the image with this information.

```python
left_lane_base = left_fit[0]*binary_warped.shape[0]**2 + left_fit[1]*binary_warped.shape[0] + left_fit[2]
right_lane_base = right_fit[0]*binary_warped.shape[0]**2 + right_fit[1]*binary_warped.shape[0] + right_fit[2]
lane_center = (left_lane_base + right_lane_base) / 2
car_position = image.shape[1] / 2
center_offset = (lane_center - car_position) * xm_per_pix

cv2.putText(result, 'Center Offset: {:.2f}m'.format(center_offset), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```

### **2.4 [Main Pipeline](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/notebook/2.4_main_pipeline.ipynb)**

The main pipeline is the core of the entire image processing sequence. It sequentially executes image preprocessing, sliding windows, and draw lane pipeline for each frame.

### **Main Pipeline Steps**

**2.4.1 Call Image Preprocessing**: Preprocess the input image to produce a binarized image.

```python
binary_warped, matrix = image_processing_pipeline(image, src_mask, dst_mask)

```

**2.4.2 Call Sliding Windows**: Use the sliding windows technique on the binarized image to detect lane lines.

```python
left_fit, right_fit, lefty, leftx, righty, rightx, out_img = sliding_windows(binary_warped)
```

**2.4.3 Use the perspective-transformed image directly and place it in the top-left corner**: Resize and place it in the top-left corner.

```python
small_warped = cv2.resize(msk, (320, 180))
```

**2.4.4 Call Draw Lane Pipeline**: Draw the detected lane lines on the original image.

```python
result = draw_lane_pipeline(image, binary_warped, matrix, left_fit, right_fit)
lane_result[0:180, 0:320] = small_warped
```

**2.4.5 Return Result Image**: Return the processed result image.

```python
return result
```

### **2.5 Results**

The system was tested on real driving conditions, and it successfully detected lane lines with high accuracy. The robustness of the approach is demonstrated through consistent performance across different scenarios.

![result](https://github.com/ChoiDoYeun/lane_detection_using_opencv/blob/main/image_readme/2.5_result.png)

## **3. Discussion**

The system demonstrates robustness in standard driving conditions. However, its performance can be affected by sharp turns, extreme lighting variations, and occlusions. Additionally, if there are painted lines on the road center, those areas may become the peak values, causing detection issues. Due to the use of dashcam footage, calibration was difficult and could not be performed effectively. Future enhancements will focus on adaptive perspective transformations and improved thresholding techniques to handle these challenges more effectively.

## **4. Conclusion**

This project presents a comprehensive approach to lane detection using computer vision techniques. By combining multiple image processing steps, the system achieves reliable lane identification and provides crucial data for autonomous vehicle navigation. In tests, the system successfully detected lanes with a high degree of accuracy under various conditions, demonstrating its robustness and applicability. Ongoing enhancements will further improve its performance in diverse driving scenarios.
