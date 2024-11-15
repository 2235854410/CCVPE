import pickle

# 加载pkl文件
with open('/home/test/code/CCVPE/dataset/vigor_gps_dict_cross_debug2.pkl', 'rb') as f:
    data = pickle.load(f)

# 查看文件中的内容
print(data)