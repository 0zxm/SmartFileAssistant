import sys
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'D:\\MyCode\\VSCode\\python\\毕设-文件自动分类语义搜索\\venv311_pytorch\\Lib\\site-packages\\PyQt5\\Qt5\\plugins'
import torch
from PyQt5.QtWidgets import (
    QApplication,
)

from ui import FileClassifierApp

def main():
    app = QApplication(sys.argv)
    ex = FileClassifierApp()
    ex.resize(800,600)
    ex.show()    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()   