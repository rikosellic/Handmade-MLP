from network import *

cfg={
    'input_dimension': 2,   #输入维度
    'hidden_dimension': 16, #隐藏层节点数
    'output_dimension': 1,  #输出维度
    'learning_rate': 0.1,  #学习率
    'momentum': 0.9  #momentum常量
}

if __name__ == '__main__':
    N=MLQP(cfg)
    N.load_and_normalize_data(train_path='two_spiral_train_data.txt',test_path='two_spiral_test_data.txt')
    N.train(20000,log_path="MLQP_16_lr0.1.txt")
    #N.plot_decision_bound_2d(HiddenW_path='MLP/epoch10000_lr5_dim16_HiddenW.npy',OutputW_path='MLP/epoch10000_lr5_dim16_OutputW.npy')
    #N.plot_decision_bound_2d(HiddenW_path='MLQP/MLQP_epoch10000_lr0.1_dim16_HiddenW.npy',HiddenW2_path='MLQP/MLQP_epoch10000_lr0.1_dim16_HiddenW2.npy', OutputW_path='MLQP/MLQP_epoch10000_lr0.1_dim16_OutputW.npy',OutputW2_path='MLQP/MLQP_epoch10000_lr0.1_dim16_OutputW2.npy')
    #plot_loss()

