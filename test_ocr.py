import cv2
import easyocr
import os

VIDEO_PATH = "TEST VIDEO.mp4"

def test_ocr():
    print("Testing OCR on video frames...")
    reader = easyocr.Reader(['en'], gpu=False)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    frame_count = 0
    while cap.isOpened() and frame_count < 1000:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        
        # Check every 100 frames
        if frame_count % 100 == 0:
            print(f"Checking frame {frame_count}...")
            results = reader.readtext(frame)
            for (bbox, text, prob) in results:
                print(f"  [OCR] Found: '{text}' (prob: {prob:.2f})")
    
    cap.release()
    print("OCR test complete.")

if __name__ == "__main__":
    test_ocr()
