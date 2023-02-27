'''从训练集中的每一类中随机抽取20%，移动到验证文件夹val对应的子类文件夹中,
 从训练集中每一类随机抽取10%，移动到文件夹test对应的子文件夹中。
'''
import os,random,shutil
path='images_original'#直接修改这里即可更改对不同数据集的预处理
train_path='train'
val_path='val'
test_path='test'
def joint(a1,a2):
    return os.path.join(a1,a2) #将a1，a2路径连接起来
source=joint(path,train_path)
destination_val=joint(path,val_path)
destination_test=joint(path,test_path)

def moveFile_val_test(dir_name):#将dir_name文件夹中的一部分放入.../val文件夹中
    dir_s = joint(source, dir_name)
    dir_d_val=joint(destination_val, dir_name)
    dir_d_test = joint(destination_test,dir_name)
    os.makedirs(dir_d_val)#建立文件夹
    os.makedirs(dir_d_test)
    rate_val = 0.2  # 自定义抽取图片的比例
    rate_test = 0.1 #测试集抽取图片比例
    pathDir=os.listdir(dir_s)#打开dir_s文件夹
    filenumber=len(pathDir)#获取该文件夹中文件数量
    picknumber_val = int(filenumber * rate_val)
    # 按照rate比例从文件夹中取一定数量图片
    sample_val = random.sample(pathDir, picknumber_val)
    # 随机选取picknumber数量的样本图片
    print(sample_val)#打印出抽取的图片，可去除
    print("-"*20)
    for name in sample_val:
        shutil.move(joint(dir_s,name), dir_d_val)#先将val提出来
    picknumber_test = int(filenumber * rate_test)#保留移动train数居前的数据量
    pathDir_again = os.listdir(dir_s)#重载 更新缓存
    sample_test = random.sample(pathDir_again, picknumber_test)
    print(sample_test)
    for name in sample_test:
        shutil.move(joint(dir_s,name), dir_d_test)

# def moveFile_test(dir_name):#将dir_name文件夹中的一部分放入.../test文件夹中
#     dir_s = joint(source, dir_name)
#     dir_d=joint(destination_test, dir_name)
#     os.makedirs(dir_d)#建立文件夹
#     rate = 0.3  # 自定义抽取图片的比例
#     pathDir=os.listdir(dir_s)#打开dir_s文件夹
#     filenumber=len(pathDir)#获取该文件夹中文件数量
#     picknumber = int(filenumber * rate)
#     # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(pathDir, picknumber)
#     # 随机选取picknumber数量的样本图片
#     print(sample)#打印出抽取的图片，可去除
#     for name in sample:
#         shutil.move(joint(dir_s,name), dir_d)

if __name__=='__main__':
    for dir1 in os.listdir(source):
        moveFile_val_test(dir1)
    # for dir2 in os.listdir(source):
    #     moveFile_test((dir2))