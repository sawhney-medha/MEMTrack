# Debugging known Issues

1. **Error with OpenCV functions**
   OpenCV(4.5.1) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-1drr4hl0\opencv\modules\highgui\src\window.cpp:651: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.

   ```bash
   pip uninstall opencv-python-headless -y
   pip install opencv-python --upgrade
   
2. **Numpy error**
ImportError: lap requires numpy, please "pip install numpy"

   ```bash
   pip install numpy

3. **Installing pycocotools**
Gthub link fails
   ```bash
   pip install pycocotools>=2.0.2
