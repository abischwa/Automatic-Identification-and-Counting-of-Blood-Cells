import cv2
import time
import os
import json
import csv
from utils import iou
from scipy import spatial
from darkflow.net.build import TFNet

options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.1,
           'gpu': 0.7}

tfnet = TFNet(options)

pred_bb = []  # predicted bounding box
pred_cls = []  # predicted class
pred_conf = []  # predicted class confidence


def blood_cell_count(file_name, save_counts=True, show_image=True):
    rbc = 0
    wbc = 0
    platelets = 0

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    tic = time.time()
    image = cv2.imread('data/' + file_name)
    
    if image is None:
        print(f"Error: Image '{file_name}' not found.")
        return None
    
    output = tfnet.return_predict(image)

    for prediction in output:
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue

        # clearing up overlapped same platelets
        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index])
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue

            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)

        if label == 'RBC':
            color = (255, 0, 0) #Blue in BGR
            rbc = rbc + 1
        if label == 'WBC':
            color = (0, 255, 0) #Green in BGR
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 0, 255) #Red in BGR
            platelets = platelets + 1

        radius = int((br[0] - tl[0]) / 2)
        image = cv2.circle(image, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[0], tl[1], br[0], br[1]])

        if label == 'RBC':
            cls.append(0)
        if label == 'WBC':
            cls.append(1)
        if label == 'Platelets':
            cls.append(2)

        conf.append(confidence)

    toc = time.time()
    pred_bb.append(cell)
    pred_cls.append(cls)
    pred_conf.append(conf)
    avg_time = (toc - tic) * 1000

    # Calculate total cells
    total_cells = rbc + wbc + platelets

    # Add count to overlay image
    font = cv2.FONT_HERSHEY_SIMPLEX
    count_text = [
        f"Total Cells: {total_cells}",
        f"RBC: {rbc}",
        f"WBC: {wbc}",
        f"Platelets: {platelets}"
    ]

    #Add semi-transparent background for text
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    #add count text to image
    for i, text in enumerate(count_text):
        y_pos = 35 + i * 25
        cv2.putText(image, text, (15, y_pos), font, 0.6, (255, 255, 255), 2)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Save the annotated image
    cv2.imwrite('output/' + file_name, image)

    #Print results to console
    print(f"\n=== Results for {file_name}===")
    print(f"processing time: {avg_time: .2f} ms")
    print(f"Total Cells: {total_cells}")
    print(f"Red Blood Cells (RBC): {rbc}")
    print(f"White Blood Cells (WBC): {wbc}")
    print(f"Platelets: {platelets}")
    print(f"Annotated image saved to 'output/{file_name}'")
    
    # Save counts to files
    if save_counts:
        save_count_data(file_name, rbc, wbc, platelets, total_cells, avg_time)

    # Show image if requested
    if show_image:
        window_title = f'Total RBC: {rbc}, WBC: {wbc}, Platelets: {platelets}'
        cv2.imshow(window_title, image)
        print('Press "ESC" to close window . . .')
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    
    # Return counts for further processing
    return {
        'file_name': file_name,
        'rbc': rbc,
        'wbc': wbc,
        'platelets': platelets,
        'total': total_cells,
        'processing_time_ms': avg_time
    }

def save_count_data(file_name, rbc, wbc, platelets, total_cells, avg_time):
    """Save the counts and processing time to a CSV file."""
    print('{0:.5}'.format(avg_time), 'ms')

    #Create a Results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save individual results to JSON
    name_without_ext = os.path.splitext(file_name)[0]
    json_file = f'results/{name_without_ext}_counts.json'
    os.makedirs(os.path.dirname(json_file), exist_ok=True) 

    result_data = {
        'image': file_name,
        'counts': {
            'RBC': rbc,
            'WBC': wbc,
            'Platelets': platelets,
            'Total Cells': total_cells
        },
        'processing_time_ms': avg_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(json_file, 'w') as f:
        json.dump(result_data, f, indent=2)

# Append results to a CSV file
    csv_file = 'results/blood_cell_counts.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            #Write header only if file does not exist
            writer.writerow(['Image', 'RBC', 'WBC', 'Platelets', 'Total Cells', 'Processing Time (ms)', 'Timestamp'])

        # Write the data
        writer.writerow([file_name, rbc, wbc, platelets, total_cells, f"{avg_time:.2f}",
                        time.strftime('%Y-%m-%d %H:%M:%S')])
        
    print(f"Count data saved to {json_file}")
    print(f"Summary updated in {csv_file}")

def process_multiple_images(image_directory='data', show_images=False):
    """Processes all images in the specified directory."""

    if not os.path.exists(image_directory):
        print(f"Error: Directory '{image_directory}' does not exist.")
        return

    # Get all image files in the directory
    image_extentions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', 'tif']
    image_files = [f for f in os.listdir(image_directory)
                   if any(f.lower().endswith(ext) for ext in image_extentions)]

    if not image_files:
        print(f"No image files found in '{image_directory}'.")
        return

    print(f"Found {len(image_files)} image files to process...")

    all_results = []
    total_summary = {'rbc': 0, 'wbc': 0, 'platelets': 0, 'total_cells': 0}

    for i, image_file in enumerate(image_files, 1):
        print(f"\nProcessing image {i}/{len(image_files)}: {image_file}")

        result = blood_cell_count(image_file, save_counts=True, show_image=show_images)

        if result:
            all_results.append(result)
            total_summary['rbc'] += result['rbc']
            total_summary['wbc'] += result['wbc']
            total_summary['platelets'] += result['platelets']
            total_summary['total_cells'] += result['total']

    # Print summary of all processed images
    print(f"\n{'='*50}")
    print(f"Processing Complete! Summary of all images:")
    print(f"{'='*50}")
    print(f"Total Images Processed: {len(all_results)}")
    print(f"Total Red Blood Cells (RBC): {total_summary['rbc']}")
    print(f"Total White Blood Cells (WBC): {total_summary['wbc']}")
    print(f"Total Platelets: {total_summary['platelets']}")
    print(f"Total Cells: {total_summary['total_cells']}")

    if all_results:
        avg_rbc = total_summary['rbc'] / len(all_results)
        avg_wbc = total_summary['wbc'] / len(all_results)
        avg_platelets = total_summary['platelets'] / len(all_results)
        avg_total_cells = total_summary['total_cells'] / len(all_results)

        print(f"\nAverage Counts per Image:")
        print(f"Average RBC: {avg_rbc:.1f}")
        print(f"Average WBC: {avg_wbc:.1f}")
        print(f"Average Platelets: {avg_platelets:.1f}")
        print(f"Average Total Cells: {avg_total_cells:.1f}")
    return all_results
 
# Main Execution
if __name__ == "__main__":
    print("Blood Cell Counting and Detection")
    print("=" * 40)
          
    # Option 1: Process a single image
    image_name = 'image_001.jpg'
    blood_cell_count(image_name)

    # Option 2: Process multiple images
    # Uncomment the following lines to process all images in the 'data' directory
    print("n\Option 2: Process all images in data folder")
    user_input = input("Do you want to process all images in the 'data' folder? (y/n): ").strip().lower()

    if user_input == 'y':
        show_each = input("show each image window? (y/n): ").lower().strip()=='y'
        process_multiple_images(show_images=show_each)

    print('\nAll Done!')