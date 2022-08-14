import os

log_path = "logs"

#   删除目标文件夹
if os.path.isdir("logs"):  # 判断是否有这个名字的文件夹
    log_path_list = os.listdir(log_path)
    #   删除目标文件夹内的文件
    for file in log_path_list:
        file_path = os.path.join(log_path, file)
        os.remove(file_path)
    #   这个函数只能删除空文件夹
    os.rmdir("logs")
    print("已删除存在的logs文件")
else:
    print("并不存在logs文件")
