# import tkinter as tk
# from tkinter.simpledialog import askstring, askinteger, askfloat
# # 接收一个整数
# def print_integer():
#   res = askinteger("Spam", "Egg count", initialvalue=12*12)
#   print(res)
# # 接收一个浮点数
# def print_float():
#   res = askfloat("Spam", "Egg weight\n(in tons)", minvalue=1, maxvalue=100)
#   print(res)
# # 接收一个字符串
# def print_string():
#   res = askstring("Spam", "Egg label")
#   print(res)
# root = tk.Tk()
# tk.Button(root, text='取一个字符串', command=print_string).pack()
# tk.Button(root, text='取一个整数', command=print_integer).pack()
# tk.Button(root, text='取一个浮点数', command=print_float).pack()
# root.mainloop()
from socket import INADDR_MAX_LOCAL_GROUP
import numpy as np
import pandas as pd
def save_pandas(a_,path):
    # a_ = [[(4,5),['1.2',"2.3","5.4"]],[(6,5),['10.2',"20.3","50.4"]]]
    X_dim = 0
    Y_dim = 1
    Z_dim = 2

    ux_matrix = np.zeros((len(a_),2))
    xyz_matrix = np.zeros((len(a_),3))

    index_ = 0
    for single_dict in a_:
        ux_matrix[index_,X_dim] = single_dict[0][X_dim]
        ux_matrix[index_,Y_dim] = single_dict[0][Y_dim]
        xyz_matrix[index_,X_dim] = float(single_dict[1][X_dim])
        xyz_matrix[index_,Y_dim] = float(single_dict[1][Y_dim])
        xyz_matrix[index_,Z_dim] = float(single_dict[1][Z_dim])
        index_ +=1
    m = np.concatenate((ux_matrix,xyz_matrix),axis=1)
    df = pd.DataFrame(m, columns=['u', 'v', 'x', 'y','z'])
    a = 1
    df.to_csv(path,index=None)

import tkinter as tk
import tkinter
from PIL import Image, ImageTk
import cv2
# global root
root = tk.Tk()
 
def create():
    """
    1 返回图像
    2 显示图像
    
    """
    img_open = cv2.imread(r"practice\1.png")


    Image_Width, Image_Height = 1080, 720

    # img_open = Image.open(r"practice\1.png")
    
    top = tk.Toplevel()
    top.geometry("%sx%s" % (Image_Width, Image_Height))
    top.title("Python")
    img_open = cv2.resize(img_open,(Image_Width, Image_Height))
    # msg = tk.Message(top, text="I love Python!")
    # msg.pack()
    img_open = cv2.cvtColor(img_open, cv2.COLOR_BGR2RGBA)
    img_open = Image.fromarray(img_open)
    img_png = ImageTk.PhotoImage(img_open)
    label_img = tkinter.Label(top, image = img_png,)
    label_img.pack()
    top.mainloop()

 
tk.Button(root, text="创建顶级窗口", command=create).pack()

root.mainloop()
print(1)
import os
import glob
def  get_files_basename(dir_path, fileExtensions):
    """获取文件夹下某几种后缀的基础名称，返回列表

    Args:
        fileExtensions (_type_): _description_
    """
    fileExtensions = [ "csv" ]
    listOfFiles    = []
    listOfFiles_basename    = []
    img_dir  = r'D:\files\temp\t'
    for extension in fileExtensions:
        listOfFiles.extend( glob.glob( img_dir + '\\*.' + extension  ))
        # listOfFiles.extend( glob.glob( img_dir + '\\*.' + extension.upper()))
    for file_ in listOfFiles:
        listOfFiles_basename.append(os.path.splitext(os.path.split(file_)[-1])[0])
    a = 1
