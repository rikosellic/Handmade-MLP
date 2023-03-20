import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s=1/(1+np.exp(-x))
    ds = s * (1-s)
    return ds

def max_min_normalize(x,upper,lower):
    s=   np.divide(np.subtract(2*x,np.add(lower,upper)),np.subtract(upper,lower))
    return s

def calculate_loss(e):
     return np.sum(np.multiply(e,e))/2

#父类Network, 提供训练框架
class Network:
    def __init__(self,cfg):
        try:
            plt.rcParams['figure.figsize']=(12.8,12.8)
            self.input_dimension :int = cfg['input_dimension']
            self.hidden_dimension :int = cfg['hidden_dimension']
            self.output_dimension :int = cfg['output_dimension']
            self.learning_rate :float= cfg['learning_rate']
            self.momentum :float= cfg['momentum']
            self.train_X = []
            self.train_Y=[]
            self.test_X=[]
            self.test_Y=[]
            self.test_X_unnormarlized=[]
            self.train_X_max=None
            self.train_X_min=None
            self.train_num=0
            self.test_num=0
            self.train_path=None
            self.test_path=None

        except KeyError as e:
            print('cfg缺失必要的项', e)

    #加载数据集, 进行max-min归一化到[-1,1]
    def load_and_normalize_data(self,train_path,test_path):
        self.train_path=train_path
        self.test_path=test_path
        #训练集
        with open(train_path,'r') as f:
            for line in f.readlines():
                line= list(map(float, filter (lambda s: s != '', line.strip().split(' '))))
                if(len(line)!=self.input_dimension+1):
                    raise ValueError("数据不符合输入类型")
                self.train_X.append(line[:2])
                self.train_Y.append(line[-1])

        self.train_num=len(self.train_X)
        self.train_Y = np.asarray(self.train_Y).reshape((self.train_num,1))
        self.train_X=np.asarray(self.train_X)
        self.train_X_min=np.min(self.train_X,axis=0)
        self.train_X_max = np.max(self.train_X, axis=0)
        self.train_X = max_min_normalize(self.train_X,lower=self.train_X_min,upper=self.train_X_max)
        self.train_X=self.train_X.T

        #根据训练集参数对测试机进行归一化
        with open(test_path,'r') as f:
            for line in f.readlines():
                line= list(map(float, filter (lambda s: s != '', line.strip().split(' '))))
                if(len(line)!=self.input_dimension+1):
                    raise ValueError("数据不符合输入类型")
                self.test_X.append(line[:2])
                self.test_Y.append(line[-1])
        self.test_X_unnormarlized=self.test_X
        self.test_num = len(self.test_X)
        self.test_Y = np.asarray(self.test_Y).reshape((self.test_num, 1))
        self.test_X = np.asarray(self.test_X)
        self.test_X = max_min_normalize(self.test_X, lower=self.train_X_min, upper=self.train_X_max)
        self.test_X = self.test_X.T


    #反向传播, 由子类重写
    def backward_propagation(self,e):
        pass

    #正向求职, 由子类重写
    def evaluate(self,x):
        pass

    #判断输出的分类与真值是否相同, 由子类重写
    def accurate(self,actual,desire):
        pass

    #保存网络权重, 由子类重写
    def save(self,epoch:int):
        pass

    #训练, 每轮训练完后在测试集上测试
    def train(self, epoch:int,log_path):
        with open(log_path,'w') as f:
            for i in range(epoch):
                train_loss=0

                for j in range(self.train_num):
                    x=self.train_X[:,j,None]
                    d=self.train_Y[j,:,None]
                    out=self.evaluate(x)
                    e=np.subtract(d,out)
                    train_loss+=calculate_loss(e)
                    self.backward_propagation(e)

                train_loss=train_loss/self.train_num

                test_loss = 0
                test_accuracy=0
                for k in range(self.test_num):
                    x = self.test_X[:, k, None]
                    d = self.test_Y[k, :, None]
                    out = self.evaluate(x)
                    e = np.subtract(d, out)
                    test_loss+=calculate_loss(e)
                    if self.accurate(actual=out,desire=d):
                        test_accuracy+=1
                test_loss=test_loss/self.test_num
                test_accuracy=test_accuracy/self.test_num
                print('Epoch: ', i+1,'train_loss:',train_loss,'test_loss',test_loss, 'accuracy', test_accuracy )
                print('Epoch: ', i + 1, 'train_loss:', train_loss, 'test_loss', test_loss, 'accuracy', test_accuracy,file=f)
                if (i+1)%10000==0:
                    self.save(i+1)

#多层感知机
class MLP(Network):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.__Input_extend = np.zeros((self.input_dimension+1,1),dtype=np.float_)              #将输入扩展一维, 以便将bias整合到权重中
        self.__HiddenW = np.random.rand(self.hidden_dimension, self.input_dimension + 1)  #隐藏层权重
        self.__HiddenV = np.zeros((self.hidden_dimension,1), dtype=np.float_)                     #隐藏层加权求和结果
        self.__HiddenY = np.zeros((self.hidden_dimension+1,1), dtype=np.float_)                   #隐藏层输出
        self.__OutputW = np.random.rand(self.output_dimension, self.hidden_dimension + 1)       #输出层权重
        self.__OutputV = np.zeros((self.output_dimension,1), dtype=np.float_)                       #输出层加权求和结果
        self.__Output = np.zeros((self.output_dimension,1), dtype=np.float_)                         #输出

        self.__OutputLocalGrad = np.zeros((self.output_dimension, 1), dtype=np.float_)
        self.__OutputDeltaW = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__OutputDeltaWOld = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__HiddenLocalGrad = np.zeros((self.hidden_dimension, 1), dtype=np.float_)
        self.__HiddenDeltaW= np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)
        self.__HiddenDeltaWOld = np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)

    def evaluate(self,x):
        one = np.ones((1, 1), dtype=np.float_)
        np.concatenate((one,x),axis=0,out=self.__Input_extend)
        np.matmul(self.__HiddenW, self.__Input_extend,out=self.__HiddenV)             #隐藏层加权求和
        np.concatenate((one,sigmoid(self.__HiddenV)),out=self.__HiddenY)                   #sigmoid激活层
        np.matmul(self.__OutputW,self.__HiddenY,out=self.__OutputV)                       #输出层加权求和
        self.__Output=sigmoid(self.__OutputV)                                                               #sigmoid激活层
        return(self.__Output)

    def accurate(self,actual,desire):
        for i in range(self.output_dimension):
            if actual[i][0]>=0.5 and desire[i][0]==0 or actual[i][0]<0.5 and desire[i][0]==1:
                return False
        return True

    def backward_propagation(self,e):
        np.multiply( sigmoid_derivative(self.__OutputV),e,out=self.__OutputLocalGrad )                     #计算输出层Local Gradient
        np.matmul( self.__OutputLocalGrad,self.__HiddenY.T,out=self.__OutputDeltaW)                   #输出层权重更新量
        self.__OutputDeltaW=self.learning_rate * self.__OutputDeltaW + self.momentum*  self.__OutputDeltaWOld    #更新输出层权重
        np.matmul(self.__OutputW.T[1:],self.__OutputLocalGrad,out=self.__HiddenLocalGrad)
        np.multiply(sigmoid_derivative(self.__HiddenV),self.__HiddenLocalGrad,out=self.__HiddenLocalGrad)  #计算隐藏层Local Gradient
        np.matmul(self.__HiddenLocalGrad, self.__Input_extend.T, out=self.__HiddenDeltaW)                         #隐藏层权重更新量
        self.__HiddenDeltaW = self.learning_rate * self.__HiddenDeltaW + self.momentum * self.__HiddenDeltaWOld  #更新隐藏层权重
        np.add(self.__OutputDeltaW,self.__OutputW,out=self.__OutputW)
        np.add(self.__HiddenDeltaW,self.__HiddenW,out=self.__HiddenW)
        self.__OutputDeltaWOld=self.__OutputDeltaW
        self.__HiddenDeltaWOld=self.__HiddenDeltaW

    def save(self,epoch):
        np.save("epoch{}_lr{}_dim{}_HiddenW.npy".format(epoch,self.learning_rate,self.hidden_dimension),self.__HiddenW)
        np.save("epoch{}_lr{}_dim{}_OutputW.npy".format(epoch,self.learning_rate,self.hidden_dimension), self.__OutputW)

    def plot_decision_bound_2d(self,HiddenW_path,OutputW_path):
        if self.input_dimension!=2 or self.output_dimension!=1:
            return
        self.__HiddenW=np.load(HiddenW_path)
        self.__OutputW=np.load(OutputW_path)
        white_dot_x=[]
        white_dot_y = []
        black_dot_x = []
        black_dot_y = []
        negative_blue_x=[]
        negative_blue_y=[]
        positive_yellow_x=[]
        positive_yellow_y=[]
        for i in range(self.test_num):
            if self.test_Y[i][0]==0:
                black_dot_x.append(self.test_X_unnormarlized[i][0])
                black_dot_y.append(self.test_X_unnormarlized[i][1])
            else:
                white_dot_x.append(self.test_X_unnormarlized[i][0])
                white_dot_y.append(self.test_X_unnormarlized[i][1])
        for x in np.arange(-6, 6, 0.02):
            for y in np.arange(-6, 6, 0.02):
                data = np.asarray([x, y], dtype=np.float_).reshape((1, 2))

                data_norm = max_min_normalize(data, upper=self.train_X_max, lower=self.train_X_min)
                data_norm = data_norm.T
                ret = self.evaluate(data_norm)
                if ret[0][0] >= 0.5:
                    negative_blue_x.append(x)
                    negative_blue_y.append(y)
                else:
                    positive_yellow_x.append(x)
                    positive_yellow_y.append(y)

        plt.title('MLP_lr5_epoch10000')
        plt.scatter(x=negative_blue_x, y=negative_blue_y, c='darkgray', s=5)
        plt.scatter(x=positive_yellow_x, y=positive_yellow_y, c='salmon', s=5)
        plt.scatter(x=black_dot_x, y=black_dot_y, c='k', s=20)
        plt.scatter(x=white_dot_x, y=white_dot_y, c='w', s=20)
        plt.show()

#多层二次感知机
class MLQP(Network):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.__Input_extend = np.zeros((self.input_dimension+1,1),dtype=np.float_)
        self.__HiddenW = np.random.rand(self.hidden_dimension, self.input_dimension + 1)
        self.__HiddenW2 = np.random.rand(self.hidden_dimension, self.input_dimension + 1)   #隐藏层二次项权重
        self.__HiddenV = np.zeros((self.hidden_dimension,1), dtype=np.float_)
        self.__HiddenY = np.zeros((self.hidden_dimension+1,1), dtype=np.float_)
        self.__OutputW = np.random.rand(self.output_dimension, self.hidden_dimension + 1)
        self.__OutputW2 = np.random.rand(self.output_dimension, self.hidden_dimension + 1) #输出层二次项权重
        self.__OutputV = np.zeros((self.output_dimension,1), dtype=np.float_)
        self.__Output = np.zeros((self.output_dimension,1), dtype=np.float_)

        self.__OutputLocalGrad = np.zeros((self.output_dimension, 1), dtype=np.float_)
        self.__OutputDeltaW = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__OutputDeltaWOld = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__OutputDeltaW2 = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__OutputDeltaW2Old = np.zeros((self.output_dimension, self.hidden_dimension + 1), dtype=np.float_)
        self.__HiddenLocalGrad = np.zeros((self.hidden_dimension, 1), dtype=np.float_)
        self.__HiddenDeltaW= np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)
        self.__HiddenDeltaWOld = np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)
        self.__HiddenDeltaW2 = np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)
        self.__HiddenDeltaW2Old = np.zeros((self.hidden_dimension, self.input_dimension + 1), dtype=np.float_)

    def evaluate(self,x):
        one = np.ones((1, 1), dtype=np.float_)
        np.concatenate((one,x),axis=0,out=self.__Input_extend)
        np.matmul(self.__HiddenW2, np.multiply(self.__Input_extend,self.__Input_extend),out=self.__HiddenV)
        np.add(self.__HiddenV,np.matmul(self.__HiddenW,self.__Input_extend),out=self.__HiddenV)
        np.concatenate((one,sigmoid(self.__HiddenV)),out=self.__HiddenY)
        np.matmul(self.__OutputW2,np.multiply(self.__HiddenY,self.__HiddenY),out=self.__OutputV)
        np.add(self.__OutputV,np.matmul(self.__OutputW,self.__HiddenY),out=self.__OutputV)
        self.__Output=sigmoid(self.__OutputV)
        return(self.__Output)

    def accurate(self,actual,desire):
        for i in range(self.output_dimension):
            if actual[i][0]>=0.5 and desire[i][0]==0 or actual[i][0]<0.5 and desire[i][0]==1:
                return False
        return True

    def backward_propagation(self,e):
        np.multiply( sigmoid_derivative(self.__OutputV),e,out=self.__OutputLocalGrad ) #输出层local gradient
        np.matmul( self.__OutputLocalGrad,self.__HiddenY.T,out=self.__OutputDeltaW)
        self.__OutputDeltaW=self.learning_rate * self.__OutputDeltaW + self.momentum*  self.__OutputDeltaWOld
        np.matmul(self.__OutputLocalGrad, np.multiply(self.__HiddenY,self.__HiddenY).T, out=self.__OutputDeltaW2)
        self.__OutputDeltaW2 = self.learning_rate * self.__OutputDeltaW2 + self.momentum * self.__OutputDeltaW2Old

        np.matmul(self.__OutputW2.T[1:], self.__OutputLocalGrad, out=self.__HiddenLocalGrad)
        np.multiply(self.__HiddenLocalGrad,2*self.__HiddenY[1:],out=self.__HiddenLocalGrad)  #隐藏层local gradient
        np.add(self.__HiddenLocalGrad,np.matmul(self.__OutputW.T[1:], self.__OutputLocalGrad),out=self.__HiddenLocalGrad)
        np.multiply(sigmoid_derivative(self.__HiddenV),self.__HiddenLocalGrad,out=self.__HiddenLocalGrad)


        np.matmul(self.__HiddenLocalGrad, self.__Input_extend.T, out=self.__HiddenDeltaW)
        self.__HiddenDeltaW = self.learning_rate * self.__HiddenDeltaW + self.momentum * self.__HiddenDeltaWOld
        np.matmul(self.__HiddenLocalGrad, np.multiply(self.__Input_extend,self.__Input_extend).T, out=self.__HiddenDeltaW2)
        self.__HiddenDeltaW2 = self.learning_rate * self.__HiddenDeltaW2 + self.momentum * self.__HiddenDeltaW2Old

        np.add(self.__OutputDeltaW,self.__OutputW,out=self.__OutputW)
        np.add(self.__HiddenDeltaW,self.__HiddenW,out=self.__HiddenW)
        np.add(self.__OutputDeltaW2, self.__OutputW2, out=self.__OutputW2)
        np.add(self.__HiddenDeltaW2, self.__HiddenW2, out=self.__HiddenW2)
        self.__OutputDeltaWOld=self.__OutputDeltaW
        self.__HiddenDeltaWOld=self.__HiddenDeltaW
        self.__OutputDeltaW2Old = self.__OutputDeltaW2
        self.__HiddenDeltaW2Old = self.__HiddenDeltaW2

    def save(self,epoch):
        np.save("MLQP_epoch{}_lr{}_dim{}_HiddenW.npy".format(epoch,self.learning_rate,self.hidden_dimension),self.__HiddenW)
        np.save("MLQP_epoch{}_lr{}_dim{}_OutputW.npy".format(epoch,self.learning_rate,self.hidden_dimension), self.__OutputW)
        np.save("MLQP_epoch{}_lr{}_dim{}_HiddenW2.npy".format(epoch, self.learning_rate, self.hidden_dimension),self.__HiddenW2)
        np.save("MLQP_epoch{}_lr{}_dim{}_OutputW2.npy".format(epoch, self.learning_rate, self.hidden_dimension),self.__OutputW2)

    def plot_decision_bound_2d(self,HiddenW_path,HiddenW2_path,OutputW_path,OutputW2_path):
        if self.input_dimension!=2 or self.output_dimension!=1:
            return
        self.__HiddenW=np.load(HiddenW_path)
        self.__OutputW=np.load(OutputW_path)
        self.__HiddenW2 = np.load(HiddenW2_path)
        self.__OutputW2 = np.load(OutputW2_path)
        white_dot_x=[]
        white_dot_y = []
        black_dot_x = []
        black_dot_y = []
        negative_blue_x=[]
        negative_blue_y=[]
        positive_yellow_x=[]
        positive_yellow_y=[]
        for i in range(self.test_num):
            if self.test_Y[i][0]==0:
                black_dot_x.append(self.test_X_unnormarlized[i][0])
                black_dot_y.append(self.test_X_unnormarlized[i][1])
            else:
                white_dot_x.append(self.test_X_unnormarlized[i][0])
                white_dot_y.append(self.test_X_unnormarlized[i][1])
        for x in np.arange(-6,6,0.02):
            for y in np.arange(-6,6,0.02):
                data=np.asarray([x,y],dtype=np.float_).reshape((1,2))

                data_norm=max_min_normalize(data,upper=self.train_X_max,lower=self.train_X_min)
                data_norm=data_norm.T
                ret=self.evaluate(data_norm)
                if ret[0][0]>=0.5:
                    negative_blue_x.append(x)
                    negative_blue_y.append(y)
                else:
                    positive_yellow_x.append(x)
                    positive_yellow_y.append(y)

        plt.title('MLQP_lr0.1_epoch10000')
        plt.scatter(x=negative_blue_x, y=negative_blue_y, c='darkgray',s=5)
        plt.scatter(x=positive_yellow_x, y=positive_yellow_y, c='salmon',s=5)
        plt.scatter(x=black_dot_x,y=black_dot_y,c='k',s=20)
        plt.scatter(x=white_dot_x, y=white_dot_y, c='w',s=20)
        plt.show()

def plot_loss():
    plt.rcParams['figure.figsize'] = (12.8, 12.8)
    coordinates=[]
    files=['MLP/MLP_16_lr0.1.txt','MLP/MLP_16_lr0.5.txt','MLP/MLP_16_lr5.txt']
    for filename in files:
        with open(filename,'r') as f:
            lines=f.readlines()
            epoch=[]
            train_loss=[]
            test_loss=[]
            for line in lines:
                line=line.strip().split()[1:6:2]
                epoch.append(int(line[0]))
                train_loss.append(float(line[1]))
                test_loss.append(float(line[2]))
            coordinates.append([epoch,train_loss,test_loss])
    plt.title('MLP training_loss')
    plt.plot(coordinates[0][0], coordinates[0][1], c='darkgray',label=' lr=0.1')
    plt.plot(coordinates[1][0], coordinates[1][1], c='skyblue',label='lr=0.5')
    plt.plot(coordinates[2][0], coordinates[2][1], c='salmon',label='lr=5')
    plt.legend()
    plt.show()