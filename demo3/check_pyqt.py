import os
import PyQt5

# 获取PyQt5安装路径
pyqt5_path = os.path.dirname(PyQt5.__file__)
print('PyQt5 Installation Path:', pyqt5_path)

# 检查Qt5目录
qt5_path = os.path.join(pyqt5_path, 'Qt5')
print('Qt5 Directory Exists:', os.path.exists(qt5_path))

if os.path.exists(qt5_path):
    print('Qt5 Directory Contents:', os.listdir(qt5_path))

# 递归查找platforms目录
print('Looking for platforms directory recursively:')
for root, dirs, files in os.walk(pyqt5_path):
    if 'platforms' in dirs:
        print('Found platforms directory at:', os.path.join(root, 'platforms'))
        print('Files in platforms directory:', os.listdir(os.path.join(root, 'platforms')))
