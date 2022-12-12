import os
import glob
import numpy as np
import pandas as pd
from tkinter.simpledialog import  askfloat, askstring

def print_float():
    """
    弹出窗口获取输入 ：浮点
    """
    res = askfloat("Spam", "Egg weight\n(in tons)", minvalue=1, maxvalue=100)
    print(res)

    
def print_string():
    """
    弹出窗口获取输入：字符串
    """
    res = askstring("Spam", "Egg label")
    print(res)
    return res


def save_pandas(a_,path):
    """_summary_

    Args:
        a_ (_type_): 成员变量，包括：像素和三维坐标
        path (_type_): 保存csv路径
    """
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
    if 1 == 2:
        for file_ in listOfFiles:
            listOfFiles_basename.append(os.path.splitext(os.path.split(file_)[-1])[0])
        return listOfFiles_basename
    return listOfFiles

def check_ip_online(IP = None):
    """检查某个ip是否在线

    Args:
        IP (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    import platform
    # 获取操作系统
    sys = platform.system()
    # IP地址
    IP = "10.194.26.240" if IP==None else IP
    print(sys)

    if sys == "Windows":
        # 打开一个管道ping IP地址
        visit_IP = os.popen('ping %s' % IP)
        # 读取结果
        result = visit_IP.read()
        # 关闭os.popen()
        visit_IP.close()
        # 判断IP是否在线
        if 'TTL' in result:
            print('服务器 在线')
            return True
        else:
            print('服务器 不在线')
            return False
    elif sys == "Linux":
        visit_IP = os.popen('ping -c 1 %s' % IP)
        result = visit_IP.read()
        visit_IP.close()
        if 'ttl' in result:
            print('服务器 在线')
            return True
        else:
            print('服务器 不在线')
            return False

    else:
        print("Error")
        return False


def check_rtsp_img(device_rtsp_addr= None, size = (1920,1080)):
    """检查读取图片是否合法

    Args:
        device_rtsp_addr (_type_, optional): _description_. Defaults to None.
        size (tuple, optional): _description_. Defaults to (1920,1080).

    Returns:
        _type_: _description_
    """
    import requests
    from PIL import Image
    import io
    r = requests.get(device_rtsp_addr)
    if len(r.content) < 100 :
        return False
    
    image = Image.open(io.BytesIO(r.content))
    # with open(img_path,'wb') as f: # 截图
    #     f.write(r.content)
    #     f.close()
    if size== image.size:
        return True
    else:
        return False
    a = 1

import re

def is_ip(str):
    """断一个字符串是否是IP地址

    Args:
        str (_type_): _description_

    Returns:
        _type_: _description_
    """
    p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if p.match(str):
        return True
    else:
        return False
 