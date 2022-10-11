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