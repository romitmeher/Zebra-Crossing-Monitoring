# Zebra Crossing Monitor

A simple application that detects zebra crossings and counts pedestrian crossings using your computer's camera or video files.

## Easy Setup Steps

1. Install Python
   - Download Python from [python.org](https://python.org)
   - Run the installer
   - Check "Add Python to PATH" during installation

2. Download this project
   - Click the green "Code" button
   - Select "Download ZIP"
   - Extract the ZIP file to any folder

3. Open Command Prompt
   - Press Windows + R
   - Type "cmd" and press Enter
   - Navigate to the project folder using:
     ```
     cd path\to\project\folder
     ```

4. Create a virtual environment
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install required packages
   ```
   pip install -r requirements.txt
   ```

6. Run the application
   ```
   python app.py
   ```

7. Open your web browser
   - Go to: http://localhost:5000
   - You'll see the application interface

## How to Use

1. Upload a Video
   - Click "Choose File" or drag a video file
   - Wait for the upload to complete
   - The video will start playing automatically

2. Watch the Results
   - Green box shows the monitoring area
   - Red line shows the crossing line
   - Yellow lines show detected zebra crossing
   - Blue boxes show detected pedestrians
   - Numbers show total crossings and current pedestrians

3. Download Data
   - Click "Download Data" to save statistics
   - Data is saved as a JSON file

## Troubleshooting

If you see any errors:
1. Make sure Python is installed correctly
2. Check if all packages are installed
3. Make sure you're in the correct folder
4. Try running the commands again

## Requirements
- Windows 10 or later
- Python 3.8 or later
- Web browser (Chrome, Firefox, or Edge)
- Camera or video files for testing 