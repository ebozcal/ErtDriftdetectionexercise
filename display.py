import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image
import numpy as np

# Load ImageNet annotations from JSON file
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# Function to display image with annotations
def display_image_with_annotations(image_path, annotations):
    # Load image
    image = Image.open(image_path)
    
    # Create figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Plot bounding boxes and segmentation masks
    for annotation in annotations:
        bbox = annotation['bbox']
        bbox_rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(bbox_rect)
        
        segmentation = annotation['segmentation']
        if segmentation:
            for polygon_points in segmentation:
                polygon_array = np.array(polygon_points).reshape(-1, 2)
                segmentation_poly = Polygon(polygon_array, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(segmentation_poly)
    
    # Show plot
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'image.jpg'  # Path to the image
annotations_for_image = [annotation for annotation in annotations if annotation['image_id'] == 'image_id_of_interest']
display_image_with_annotations(image_path, annotations_for_image)
